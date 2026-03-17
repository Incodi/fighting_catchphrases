#!/usr/bin/env python
from __future__ import print_function, division

"""
Simple signature word/catchphrase finder.

Goal:
Find words/phrases a focus channel uses more than the rest.

Ultimate goal:
Find the weirdest channel. Who has the weirdest vocabulary? Who is the most unique?

"""

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import webbrowser
from collections import Counter
from datetime import datetime
from html import escape as html_escape

############ HELPERS ############

def hms_to_seconds(value):
    if not value:
        return 0.0
    parts = value.split(":")
    parts = [float(x) for x in parts]
    parts = list(reversed(parts))
    total = 0.0
    multipliers = [1, 60, 3600]
    for i in range(len(parts)):
        total += parts[i] * multipliers[i]
    return total


def extract_video_id(filename):
    match = re.search(r"\[([A-Za-z0-9_-]{11})\]", filename)
    if match:
        return match.group(1)
    return filename.replace(".txt", "").split(".")[-1]


def tokenize(text):
    if text is None:
        return []
    cleaned = re.sub(r"([^\w\s']|'(?!\w)|\s'+|'+\s)", " ", text)
    return cleaned.lower().split()

############ GET DATA ############

def load_metadata(base_dir, channel):
    path = os.path.join(base_dir, "data", "input", channel, "metadata.json")
    if not os.path.exists(path):
        return {}

    try:
        with io.open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    result = {}
    for entry in payload:
        if isinstance(entry, dict) and "id" in entry:
            result[entry["id"]] = entry
    return result


def list_text_files(base_dir, channel):
    txt_dir = os.path.join(base_dir, "data", "input", channel, "txt_files")
    if not os.path.isdir(txt_dir):
        return []

    files = []
    for name in os.listdir(txt_dir):
        if name.endswith(".txt"):
            files.append(os.path.join(txt_dir, name))
    files.sort()
    return files


def collect_tokens_for_channel(base_dir, channel, args):
    txt_files = list_text_files(base_dir, channel)
    if not txt_files:
        return None

    metadata = load_metadata(base_dir, channel)
    min_duration = hms_to_seconds(args.duration_from)
    max_duration = hms_to_seconds(args.duration_to)

    selected = []
    for path in txt_files:
        name = os.path.basename(path)
        video_id = extract_video_id(name)
        meta = metadata.get(video_id, {})

        upload_date = str(meta.get("upload_date", "99999999"))
        if args.date_from and upload_date < args.date_from:
            continue
        if args.date_to and upload_date > args.date_to:
            continue

        duration = meta.get("duration") or 0
        if min_duration and duration < min_duration:
            continue
        if max_duration and duration > max_duration:
            continue

        if args.exclude_live:
            if meta.get("was_live") == "True" or meta.get("is_live") == "True":
                continue

        selected.append(path)

    tokens = []
    files_used = 0
    for path in selected:
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print("Warning reading {0}: {1}".format(path, e))
            continue

        file_tokens = tokenize(text)
        if len(file_tokens) < args.min_tokens_per_file:
            continue

        tokens.extend(file_tokens)
        files_used += 1

    if args.token_limit > 0 and len(tokens) > args.token_limit:
        tokens = tokens[-args.token_limit:]

    if len(tokens) < args.min_tokens_total:
        return None

    return {
        "channel": channel,
        "tokens": tokens,
        "token_count": len(tokens),
        "files_used": files_used,
    }

############ COUNT NGRAMS ############

def count_ngrams(tokens, n):
    counts = Counter()
    limit = len(tokens) - n + 1
    if limit <= 0:
        return counts

    for i in range(limit):
        phrase = " ".join(tokens[i:i + n])
        counts[phrase] += 1
    return counts

############ (the meat) DETERMINE SIGNATURE PHRASES ############

def compute_scores(focus_counts, focus_total, rest_counts, rest_total, min_count, alpha):
    scores = []

    candidates = []
    for phrase, count in focus_counts.items():
        if count >= min_count:
            candidates.append(phrase)

    vocab_size = max(1, len(candidates))

    for phrase in candidates:
        f_focus = focus_counts.get(phrase, 0)
        f_other = rest_counts.get(phrase, 0)

        # This is basically the same as log-odds ratio with uninformative Dirichlet prior, 
        # as described in Monroe et al 2008 "Fightin' Words"

        p_focus = (f_focus + alpha) / (focus_total + alpha * vocab_size)
        p_rest = (f_other + alpha) / (rest_total + alpha * vocab_size)

        log_odds = math.log(p_focus) - math.log(p_rest)
        variance = (1.0 / (f_focus + alpha)) + (1.0 / (f_other + alpha))
        z_score = log_odds / math.sqrt(variance)

        dominance = 0.0
        if (f_focus + f_other) > 0:
            dominance = float(f_focus) / float(f_focus + f_other)

        scores.append({
            "phrase": phrase,
            "focus_count": f_focus,
            "rest_count": f_other,
            "log_odds": log_odds,
            "z_score": z_score,
            "dominance": dominance,
            "exclusive": (f_other == 0),
        })

    return scores

############ THE HTML EXPORT ############

def build_html(results_by_focus, focus_channels, token_counts, other_channel_count, other_token_count, args):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []
    for channel in focus_channels:
        channel_result = results_by_focus.get(channel, {})
        channel_tables = []

        ns = sorted(channel_result.keys())
        for n in ns:
            rows = channel_result[n]
            if not rows:
                continue

            label = "{0}-grams".format(n)
            if n == 1:
                label = "words"
            elif n == 2:
                label = "bigrams"
            elif n == 3:
                label = "trigrams"

            row_html = []
            rank = 1
            for item in rows:
                badge = ""
                if item["exclusive"]:
                    badge = "<span style='color:#cc6600;font-weight:700'>[EXCLUSIVE]</span> "

                row_html.append(
                    "<tr>"
                    "<td>{rank}</td>"
                    "<td>{phrase}</td>"
                    "<td style='text-align:right'>{focus}</td>"
                    "<td style='text-align:right'>{rest}</td>"
                    "<td style='text-align:right'>{log_odds:.3f}</td>"
                    "<td style='text-align:right'>{z:.2f}</td>"
                    "</tr>".format(
                        rank=rank,
                        phrase=badge + html_escape(item["phrase"]),
                        focus=item["focus_count"],
                        rest=item["rest_count"],
                        log_odds=item["log_odds"],
                        z=item["z_score"],
                    )
                )
                rank += 1

            table_html = (
                "<h3>{0}</h3>"
                "<table>"
                "<thead><tr>"
                "<th>#</th><th>Phrase</th><th>Focus #</th><th>Rest #</th><th>Log-odds</th><th>Z-score</th>"
                "</tr></thead>"
                "<tbody>{1}</tbody></table>"
            ).format(label, "".join(row_html))

            channel_tables.append(table_html)

        sections.append(
            "<section class='card'>"
            "<h2>{channel} <span class='sub'>{tokens} tokens</span></h2>"
            "{tables}"
            "</section>".format(
                channel=html_escape(channel),
                tokens=token_counts.get(channel, 0),
                tables="".join(channel_tables) if channel_tables else "<p>No results.</p>",
            )
        )

    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Fighting words</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;max-width:1100px;margin:18px auto;padding:0 14px;}"
        "h1{margin:0 0 8px 0;}"
        ".meta{font-size:13px;color:#666;margin-bottom:14px;}"
        ".card{border:1px solid #ddd;border-radius:10px;padding:12px 14px;margin-bottom:20px;background:#fafafa;}"
        "h2{margin:0 0 10px 0;font-size:18px;}"
        ".sub{font-size:12px;color:#666;font-weight:normal;}"
        "table{border-collapse:collapse;width:100%;margin-bottom:14px;}"
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;}"
        "thead{background:#f1f1f1;}"
        "</style></head><body>"
        "<h1>Fighting Words</h1>"
        "<div class='meta'>Generated {now} · Other pool: {other_channels} channels, {other_tokens} tokens</div>"
        "<div class='meta'>ngram_max={ngram_max} · min_count={min_count} · top_n={top_n} · alpha={alpha}</div>"
        "{sections}"
        "</body></html>"
    ).format(
        now=html_escape(now),
        other_channels=other_channel_count,
        other_tokens=other_token_count,
        ngram_max=args.ngram_max,
        min_count=args.min_count,
        top_n=args.top_n,
        alpha=args.alpha,
        sections="".join(sections),
    )

    return html_text

############ ARGUMENT PARSING ############

def parse_args():
    parser = argparse.ArgumentParser(description="Beginner-friendly signature phrase finder")
    parser.add_argument("channels", nargs="+", help="Channel folder names")
    parser.add_argument("--focus", nargs="+", required=True, help="Focus channels")

    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.01)

    parser.add_argument("--output-html", default="temps/discover_signature_phrases.html")

    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--duration-from", default="")
    parser.add_argument("--duration-to", default="")
    parser.add_argument("--exclude-live", action="store_true")
    parser.add_argument("--token-limit", type=int, default=0)
    parser.add_argument("--min-tokens-per-file", type=int, default=0)
    parser.add_argument("--min-tokens-total", "--min-words", dest="min_tokens_total", type=int, default=0)

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring weird unknown args: {0}".format(" ".join(unknown)))
    return args


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.abspath(os.path.join(base_dir, args.output_html))
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for focus_channel in args.focus:
        if focus_channel not in args.channels:
            args.channels.append(focus_channel)

    n_values = list(range(1, args.ngram_max + 1))

    print("getting channels...")
    channel_tokens = {}
    channel_counts = {}
    for channel in args.channels:
        data = collect_tokens_for_channel(base_dir, channel, args)
        if data is None:
            continue

        channel_tokens[channel] = data["tokens"]
        channel_counts[channel] = data["token_count"]
        print("  {0}: {1} tokens".format(channel, data["token_count"]))

    usable_focus = [c for c in args.focus if c in channel_tokens]
    if not usable_focus:
        print("The focus channel does not have any useable data.")
        return 1

    results_by_focus = {}
    other_channel_count = 0
    other_token_count = 0

    for focus_channel in usable_focus:
        focus_tokens = channel_tokens[focus_channel]
        focus_total = len(focus_tokens)

        other_tokens = []
        local_other_channels = 0
        for channel in args.channels:
            if channel == focus_channel:
                continue
            if channel not in channel_tokens:
                continue
            other_tokens.extend(channel_tokens[channel])
            local_other_channels += 1

        if local_other_channels == 0:
            print("Skipping {0}: no comparison channels with data".format(focus_channel))
            continue

        rest_total = len(other_tokens)
        other_channel_count = max(other_channel_count, local_other_channels)
        other_token_count = max(other_token_count, rest_total)

        by_n = {}
        for n in n_values:
            focus_counts = count_ngrams(focus_tokens, n)
            rest_counts = count_ngrams(other_tokens, n)

            scores = compute_scores(
                focus_counts=focus_counts,
                focus_total=focus_total,
                rest_counts=rest_counts,
                rest_total=rest_total,
                min_count=args.min_count,
                alpha=args.alpha,
            )

            scores.sort(key=lambda x: x["z_score"], reverse=True)
            by_n[n] = scores[:args.top_n]

            if by_n[n]:
                top = by_n[n][0]
                print(
                    "  {0} top {1}-gram: '{2}' (z={3:.2f}, {4} vs {5})".format(
                        focus_channel,
                        n,
                        top["phrase"],
                        top["z_score"],
                        top["focus_count"],
                        top["rest_count"],
                    )
                )

        results_by_focus[focus_channel] = by_n

    if not results_by_focus:
        print("No results to write.")
        return 1

    html_text = build_html(
        results_by_focus=results_by_focus,
        focus_channels=usable_focus,
        token_counts=channel_counts,
        other_channel_count=other_channel_count,
        other_token_count=other_token_count,
        args=args,
    )

    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    print("Wrote: {0}".format(output_path))

    try:
        opened = webbrowser.open("file://" + output_path)
        if (not opened) and sys.platform == "darwin":
            subprocess.call(["open", output_path])
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
