import os
import time, datetime
import multiprocessing
import collections
import itertools
import functools
import argparse
import pandas as pd
import pysam
import pickle


def parse_trnas(path):
    """construct table of mature tRNAs and pre-tRNA loci"""
    # read tRNA table associating transcript, anticodons, pre-tRNA loci, and
    # amino acid
    meta_pre = pd.read_csv(
        os.path.join(path, "db-trnatable.txt"),
        sep="\t",
        names=["transcript", "locus", "amino", "anticodon"],
    )
    meta_pre["locus"] = meta_pre["locus"].apply(lambda x: x.split(","))
    meta_pre = meta_pre.explode("locus", ignore_index=True)

    # read BED file for pre-tRNA loci
    bed_pre = pd.read_csv(
        os.path.join(path, "db-trnaloci.bed"),
        sep="\t",
        names=[
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "blockCount",
            "blockSizes",
            "blockStarts",
        ],
    )
    bed_pre = bed_pre[["chrom", "chromStart", "chromEnd", "name", "strand"]]

    # make additional entry for each mature tRNA transcript
    meta_mature = (
        meta_pre.groupby("transcript")[["amino", "anticodon"]].first().reset_index()
    )
    meta_mature["locus"] = meta_mature["transcript"]

    # read BED file for mature tRNAs
    bed_mature = pd.read_csv(
        os.path.join(path, "db-maturetRNAs.bed"),
        sep="\t",
        names=["chrom", "chromStart", "chromEnd", "name", "score", "strand"],
    )
    bed_mature = bed_mature[["chrom", "chromStart", "chromEnd", "name", "strand"]]

    # concatenate pre-tRNAs and mature tRNAs
    trnas = pd.concat(
        [
            meta_pre.join(bed_pre.set_index("name"), on="locus").assign(type="pre"),
            meta_mature.join(bed_mature.set_index("name"), on="locus").assign(
                type="mature"
            ),
        ],
        axis=0,
    ).reset_index(drop=True)

    return trnas


def calculate_errors(aligned_pairs, query_sequence):
    return tuple(
        (rpos, None) if qpos is None else
        (rpos, query_sequence[qpos])
        for qpos, rpos, rbase in aligned_pairs
        if qpos is None or (rbase != query_sequence[qpos] and query_sequence[qpos] != "N")
    )


def assign_mapping(read, trnas):
    """if a read alignment overlaps with a tRNA feature, represent it as the
    mapped positions of its 5' and 3' ends"""
    if (
        (read.reference_name == trnas.chrom)
        & (read.reference_start < trnas.chromEnd)
        & (read.reference_end > trnas.chromStart)
    ).any():
        errors = calculate_errors(read.aligned_pairs, read.query_sequence)
        if read.is_forward:
            return (read.reference_name, read.reference_start, read.reference_end, errors)
        else:
            # -1 since pysam uses half-open intervals to represent the forward
            # strand mapping: [reference_start, reference_end)
            return (
                read.reference_name,
                read.reference_end - 1,
                read.reference_start - 1,
                errors,
            )
    return "__no_feature"


"""relevant fields for a read mapping"""
AlignedSegment = collections.namedtuple(
    "AlignedSegment",
    ["query_name", "reference_name", "reference_start", "reference_end", "is_forward", "aligned_pairs", "query_sequence"],
)


def convert(r):
    """convert pysam AlignedSegment to AlignedSegment named tuple for pickling"""
    return AlignedSegment(
        r.query_name,
        r.reference_name,
        r.reference_start,
        r.reference_end,
        not r.is_reverse,
	tuple(r.get_aligned_pairs(with_seq=True)),
	r.query_sequence,
    )


def process_read(alns, trnas):
    """represent a list of alignments for a single read as the set of mapped
    positions of its 5' and 3' ends"""
    result = set()
    for r in alns:
        result.add(assign_mapping(r, trnas))
    return frozenset(result)


def count_reads(path, trnas, nproc=1):
    """count tRNA fragment reads in a BAM file sorted by read name. returns
    counts for each tRNA fragment class (i.e., reads whose 5' and 3' ends map
    to the same sets of positions) as a dict."""
    counts = {}
    n_reads = 0

    def increment(k):
        nonlocal n_reads

        if k in counts:
            counts[k] += 1
        else:
            counts[k] = 1
        n_reads += 1
        if n_reads % 500000 == 0:
            print(f"{datetime.datetime.now()} processed {n_reads} reads")

    t0 = time.perf_counter()

    with pysam.AlignmentFile(path, "rb") as bam:
        with multiprocessing.Pool(nproc) as pool:
            # group bam file read mappings by read name
            iter_reads = itertools.groupby(
                bam.fetch(until_eof=True), key=lambda x: x.query_name
            )
            # convert read mappings to named tuples for pickling
            iter_reads = map(lambda x: [convert(r) for r in x[1]], iter_reads)
            # process read mappings in parallel and iterate over resulting sets
            # of tRNA features
            for res in pool.imap_unordered(
                functools.partial(process_read, trnas=trnas), iter_reads, chunksize=1000
            ):
                increment(res)

    t1 = time.perf_counter()
    print("Done counting")
    print(
        f"Processed {n_reads} reads in {t1 - t0:.2f} seconds ({n_reads / (t1 - t0):.2f} reads/s)"
    )

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", help="path to tRAX tRNA genome")
    parser.add_argument("--nproc", type=int, default=1, help="number of processes")
    parser.add_argument("bam", help="BAM file")
    parser.add_argument("output", type=str, help="path to output .pkl file")
    args = parser.parse_args()

    print(args.bam, args.output)

    trnas = parse_trnas(args.genome)

    filename = os.path.basename(args.bam)
    name, ext = os.path.splitext(filename)

    counts = count_reads(args.bam, trnas, nproc=args.nproc)

    with open(args.output, "wb") as f:
        pickle.dump(counts, f)

    print("done.")
