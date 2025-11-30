#!/usr/bin/env python3
"""
Distributed Boruvka-style MST with MPI (mpi4py).

Usage:
    mpirun -n <P> python mpi_mst.py edges.txt N_vertices [--block] [--debug]

Notes:
- edges.txt: lines "u v w" (0-indexed vertices).
- N_vertices: integer number of vertices.
- By default edges are distributed round-robin by line index among ranks.
  Use --block to partition edges by contiguous blocks (block of lines).
"""

import sys
import math
import argparse
from collections import defaultdict
from mpi4py import MPI

INF = 10**30

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('edgefile', help='edge list file: u v w per line (0-indexed)')
    parser.add_argument('n', type=int, help='number of vertices')
    parser.add_argument('--block', action='store_true',
                        help='partition edges into contiguous block by file offset instead of round-robin')
    parser.add_argument('--debug', action='store_true', help='print debug logs on rank 0')
    return parser.parse_args()

def load_partitioned_edges(filename, rank, size, block=False):
    """
    Each process reads the file and keeps only its share of edges.
    If block=False -> round-robin by line index (i % size == rank)
    If block=True  -> contiguous block of approximately equal number of lines.
    """
    # First pass: count lines if block partition needed
    if block:
        total_lines = 0
        with open(filename, 'r') as f:
            for _ in f:
                total_lines += 1
        lines_per_rank = total_lines // size
        extra = total_lines % size
        start = rank * lines_per_rank + min(rank, extra)
        end = start + lines_per_rank + (1 if rank < extra else 0)
    edges = []
    with open(filename, 'r') as f:
        if not block:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i % size != rank:
                    continue
                u,v,w = line.split()
                edges.append((int(u), int(v), float(w)))
        else:
            for i, line in enumerate(f):
                if i < start or i >= end:
                    continue
                line = line.strip()
                if not line:
                    continue
                u,v,w = line.split()
                edges.append((int(u), int(v), float(w)))
    return edges

def run_distributed_boruvka(edgefile, n, block=False, debug=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0 and debug:
        print(f"[rank0] MPI size {size}; loading edges from {edgefile}")

    edges = load_partitioned_edges(edgefile, rank, size, block=block)
    if debug:
        print(f"[rank {rank}] loaded {len(edges)} edges")

    # component id per vertex (replicated on all ranks)
    comp = list(range(n))
    # final MST edges collected on rank 0
    mst_edges = []

    # helper to check number of components
    def count_components_local(comp_arr):
        return len(set(comp_arr))

    it = 0
    while True:
        it += 1
        # Each process computes local best outgoing edge for each component it sees
        local_best = {}  # comp_id -> (weight, u, v)
        for (u,v,w) in edges:
            cu = comp[u]
            cv = comp[v]
            if cu == cv:
                continue
            # candidate for cu
            cur = local_best.get(cu)
            if cur is None or w < cur[0]:
                local_best[cu] = (w, u, v)
            # candidate for cv
            cur = local_best.get(cv)
            if cur is None or w < cur[0]:
                local_best[cv] = (w, u, v)

        # Convert to list for communication
        local_candidates = [(c, tup[0], tup[1], tup[2]) for (c,tup) in local_best.items()]

        # gather all candidate lists at all ranks
        all_candidates = comm.allgather(local_candidates)  # list of lists

        # rank 0 merges to global best per component
        global_best = {}
        for proc_list in all_candidates:
            for (c, w, u, v) in proc_list:
                existing = global_best.get(c)
                if existing is None or w < existing[0]:
                    global_best[c] = (w, u, v)

        # Build a set of edges chosen to be added this round (unique by endpoint pair)
        # We deduplicate by normalized (min,max,weight)
        chosen_edges_set = {}
        for c,(w,u,v) in global_best.items():
            key = (min(u,v), max(u,v), w)
            chosen_edges_set[key] = (u,v,w)

        chosen_edges = list(chosen_edges_set.values())

        # If no chosen edges globally, algorithm is finished
        # (no outgoing edge from any component -> graph disconnected or singletons)
        # Determine globally whether there are any chosen edges
        num_chosen_local = len(chosen_edges)
        num_chosen_global = comm.allreduce(num_chosen_local, op=MPI.SUM)
        if rank == 0 and debug:
            print(f"[rank0][iter {it}] chosen edges global = {num_chosen_global}")

        if num_chosen_global == 0:
            break

        # Rank 0 will compute new component mapping by unioning endpoints of chosen edges
        # We perform union-find on current component ids to contract components.
        # Note: components are integers in [0..n-1] but many may be unused; that's fine.
        if rank == 0:
            uf = UnionFind(n)
            # union the components that are connected by chosen edges
            for (u,v,w) in chosen_edges:
                cu = comp[u]
                cv = comp[v]
                if cu != cv:
                    uf.union(cu, cv)
            # build new mapping: map old comp id -> new representative id (contiguous)
            rep_to_new = {}
            next_id = 0
            new_comp = [0]*n
            for v in range(n):
                root = uf.find(comp[v])
                if root not in rep_to_new:
                    rep_to_new[root] = next_id
                    next_id += 1
                new_comp[v] = rep_to_new[root]
            # Also prepare list of edges to add to MST: only edges that connect different components (old comp)
            # and that were part of chosen_edges. We add them to MST_edges if they actually merged two distinct components.
            add_to_mst = []
            for (u,v,w) in chosen_edges:
                if comp[u] != comp[v]:
                    add_to_mst.append((u,v,w))
            if debug:
                print(f"[rank0][iter {it}] merging -> new_component_count {next_id}; adding {len(add_to_mst)} edges to MST")
        else:
            new_comp = None
            add_to_mst = None

        # broadcast new_comp to all ranks
        new_comp = comm.bcast(new_comp, root=0)
        add_to_mst = comm.bcast(add_to_mst, root=0)

        # update comp on all ranks
        comp = new_comp

        # collect MST edges on rank 0
        if rank == 0:
            mst_edges.extend(add_to_mst)

        # Stopping condition: if only one component remains globally -> done
        unique_local = set(comp)
        num_unique_local = len(unique_local)
        # Gather minimum across ranks of unique counts is not helpful; just compute global unique count via root(0)
        # But here comp is replicated identical across ranks, so simply check on any rank
        num_components = len(set(comp))
        if rank == 0 and debug:
            print(f"[rank0][iter {it}] components remaining = {num_components}")
        if num_components <= 1:
            break

    # done
    # rank 0 prints/writes MST
    if rank == 0:
        if debug:
            print(f"[rank0] finished after {it} iterations; MST edges = {len(mst_edges)}")
        # Optionally sort output
        mst_edges.sort(key=lambda e: (e[2], e[0], e[1]))
        # print to stdout
        print("# MST edges (u v w):")
        for (u,v,w) in mst_edges:
            print(f"{u} {v} {w}")

def main():
    args = parse_args()
    run_distributed_boruvka(args.edgefile, args.n, block=args.block, debug=args.debug)

if __name__ == "__main__":
    main()

