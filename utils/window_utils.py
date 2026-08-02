from collections import defaultdict


def merge_candidates(
    candidate_rows,
    max_gap=2.0,
):
    """
    Merge neighbouring candidate chunks from the same file.

    Example

    Chunk:
    10-13
    11-14
    12-15

    becomes

    Region:
    10-15
    """

    grouped = defaultdict(list)

    # Group by filename
    for row in candidate_rows:
        grouped[row["file"]].append(row)

    merged_regions = []

    for filename, rows in grouped.items():

        rows = sorted(
            rows,
            key=lambda x: x["start_time"],
        )

        current = [rows[0]]

        for row in rows[1:]:

            previous = current[-1]

            gap = (
                row["start_time"]
                - previous["end_time"]
            )

            if gap <= max_gap:

                current.append(row)

            else:

                merged_regions.append({

                    "file": filename,

                    "rows": current,

                    "start_time": current[0]["start_time"],

                    "end_time": current[-1]["end_time"],

                })

                current = [row]

        merged_regions.append({

            "file": filename,

            "rows": current,

            "start_time": current[0]["start_time"],

            "end_time": current[-1]["end_time"],

        })

    return merged_regions