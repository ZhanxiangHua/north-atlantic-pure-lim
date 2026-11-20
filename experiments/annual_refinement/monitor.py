"""Write an independently timestamped, append-only progress history during the campaign."""
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from records import WORK, record

from tc_emulator.common import atomic


def update():
    now = datetime.now(timezone.utc)
    counters = Counter()
    efile = WORK / "experiment_records/events.jsonl"
    with efile.open() as events:
        for line in events:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            counters[item["event"]] += 1
    nadd = sum(
        1
        for d in (WORK / "data/prepared").iterdir()
        for y in d.iterdir()
        if not y.is_symlink() and (y / "complete.json").is_file()
    )
    ocean = sum(1 for _ in (WORK / "ocean_only").glob("*/*/complete.json"))
    infill = sum(1 for _ in (WORK / "infill/prepared").glob("*/*/complete.json"))
    row = dict(
        utc=now.isoformat(timespec="seconds"),
        new_paired_member_years=nadd,
        new_ocean_member_years=ocean,
        paired_cv_fits=counters["paired_refit_done"],
        full_candidate_months=counters["coupled_done"],
        budget_fits=counters["transfer_budget_done"],
        weather_sweeps=counters["weather_sweep_done"],
        completed_feature_banks=counters["feature_bank_done"],
        errors=counters["paired_refit_error"],
    )
    history = WORK / "experiment_records/progress.csv"
    first = not history.exists()
    with history.open("a") as out:
        writer = csv.DictWriter(out, list(row))
        if first:
            writer.writeheader()
        writer.writerow(row)
    extra = dict(
        utc=row["utc"],
        new_paired=nadd,
        ocean_only=ocean,
        additional_wind=infill,
        learning_curve_fits=counters["learning_curve_refit"],
        ocean_spread=counters["ocean_spread_done"],
        modal_wind=counters["modal_weather_done"],
        spatial_checks=counters["spatial_verification_done"],
        physical_verifications=counters["model_verification_done"],
        storm_cases=counters["storm_case_done"],
    )
    detail_path = WORK / "experiment_records/extended_progress.csv"
    new_file = not detail_path.exists()
    with detail_path.open("a") as sink:
        wr = csv.DictWriter(sink, list(extra))
        if new_file:
            wr.writeheader()
        wr.writerow(extra)
    body = (
        "# 实验进度\n\n"
        + f"最近记录为 {row['utc']}。目标时段为 07:12–14:12 UTC。\n\n"
        + f"新增的完整 MPI 成员年已准备 {nadd} / 597 个，其中 525 个用于训练，72 个属于新测试面板。"
        + f"海洋专用资料已准备 {ocean} / 946 个成员年，含与原始预处理逐格核对的一个控制年份。"
        + f"已为其中 {infill} / 945 个成员年补齐逐日风和大气，使训练资料可以从 30 个增至 57 个年份。\n\n"
        + f"已完成 {row['paired_cv_fits']} 轮配对交叉拟合，{row['full_candidate_months']} 个耦合模型月份实验，"
        + f"{row['budget_fits']} 个背景方差传递实验，以及 {row['weather_sweeps']} 批时间记忆实验。\n\n"
        + f"已完成 {extra['learning_curve_fits']} 个固定评估样本的训练资料量对照，以及 {extra['storm_cases']} 个使用相同扰动的气旋驱动重放。\n\n"
        + "[实验判断和数值](JOURNAL.md) · [逐事件日志](experiment_records/events.jsonl) · [进度历史](experiment_records/progress.csv) · [扩展实验记录](experiment_records/extended_progress.csv)\n"
    )
    with atomic(WORK / "STATUS.md") as dest:
        dest.write_text(body)
    return row


if __name__ == "__main__":
    spec = json.loads(Path(__file__).with_name("design.json").read_text())
    end = datetime.fromisoformat(spec["end_utc"].replace("Z", "+00:00")).timestamp()
    while True:
        row = update()
        print(row, flush=True)
        if time.time() >= end:
            break
        time.sleep(min(max(end - time.time(), 0), 600.0))
    record("monitor_end", utc=row["utc"])
