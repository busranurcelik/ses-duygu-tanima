"""Salt okunur kaynak secimi, lisans durumu ve cakisma denetimi.

Bu script:
  - CREMA-D / RAVDESS / SAVEE / TESS icinde ham_veri_listesi.csv'de geçmeyen
    orijinal WAV'lari bulur,
  - dosya icerik SHA-256 ve yol SHA-256 degerlerini egitimde kullanilan
    kayitlarla karsilastirir,
  - kaynak/sinif bazinda 15'ER kayit deterministik olarak secer,
  - secim manifesti, aday havuzu, lisans/cakisma raporu yazar.

Model, scaler, GUI veya egitim artifact'lerini degistirmez; prediction
calistirmaz. Ciktilarin tamami bu script'in bulundugu dizine yazilir.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[1]
CLASSES = ["angry", "fear", "happy", "sad"]
TARGET_PER_SOURCE_CLASS = 15

SOURCE_FOLDERS = {
    "CREMA-D": "data/CREMA-D",
    "RAVDESS": "data/RAVDESS",
    "SAVEE": "data/SAVEE",
    "TESS": "data/TESS",
}
TRAIN_SOURCE_CODE = {
    "CREMA-D": "CREMA",
    "RAVDESS": "RAVDESS",
    "SAVEE": "SAVEE",
    "TESS": "TESS",
}

PROTECTED_FILES = [
    "arayuz_A1_2_adapter.py",
    "7_final_arayuz.py",
    "test_arayuz_A1_2.py",
    "A1_2_RF_B_GUI_entegrasyon_raporu.md",
    "model_secimi_paketleme_oncesi_inceleme.md",
    "A1_2/70_15_15/modeller/rf_70_15_15_matched_A1_2.joblib",
    "A1_2/70_15_15/modeller/scaler_A1_2.joblib",
    "A1_2/70_15_15/f0_imputasyon_degerleri.csv",
    "A1_2/70_15_15/ysa/70_15_15_label_map_A1_2.json",
]

LICENSE_ROWS = [
    {
        "kaynak": "CREMA-D",
        "durum": "DOGRULANDI",
        "kapsam": "Veritabani ODbL 1.0; bireysel icerikler DbCL 1.0",
        "kanit": "https://github.com/CheyneyComputerScience/CREMA-D/blob/master/LICENSE.txt",
        "not": "GitHub README ve LICENSE.txt metni ayni kosullari belirtiyor.",
    },
    {
        "kaynak": "RAVDESS",
        "durum": "DOGRULANDI",
        "kapsam": "CC BY-NC-SA 4.0",
        "kanit": "https://zenodo.org/records/1188976",
        "not": "Zenodo kayit 1188976 metadata license alanindan okundu; ticari olmayan ve paylas-a-like kosulu var.",
    },
    {
        "kaynak": "SAVEE",
        "durum": "DOGRULANAMADI",
        "kapsam": "Belirtilmedi",
        "kanit": "-",
        "not": "Yerelde lisans/riza belgesi yok; kurumsal kaynak adresleri 404 donuyor. Ucuncu taraf repolar hak kaynagi sayilmadi.",
    },
    {
        "kaynak": "TESS",
        "durum": "DOGRULANDI",
        "kapsam": "CC BY-NC 4.0",
        "kanit": "https://doi.org/10.5683/SP2/E8H2MF",
        "not": "U of T Borealis dataset metadata license alanindan okundu; mevcut 70/15/15 test bolumunun kaynagidir.",
    },
    {
        "kaynak": "BNC",
        "durum": "HARIC",
        "kapsam": "Riza belgelenmedi",
        "kanit": "-",
        "not": "100 kayit tek konusmaciya ait (BNC_SPEAKER_01); degerlendirmeye alinmaz, sadece raporlanir.",
    },
]


def normalize_rel(value: str) -> str:
    s = str(value).replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def path_hash16(rel: str) -> str:
    return sha256_text(rel)[:16]


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        rate = int(wf.getframerate())
        channels = int(wf.getnchannels())
        frames = int(wf.getnframes())
        width = int(wf.getsampwidth())
    return {
        "sample_rate": rate,
        "channels": channels,
        "duration_s": round(frames / rate, 4) if rate else 0.0,
        "sample_width": width,
    }


def parse_label_and_speaker(source: str, rel: str) -> tuple[str | None, str | None]:
    name = Path(rel).name
    if source == "CREMA-D":
        m = re.match(r"(\d{4})_[A-Z]{3}_(ANG|FEA|HAP|SAD)_[A-Z]{2}\.wav$", name, re.I)
        if not m:
            return None, None
        label = {"ANG": "angry", "FEA": "fear", "HAP": "happy", "SAD": "sad"}[m.group(2).upper()]
        return label, "CREMA_" + m.group(1)
    if source == "RAVDESS":
        m = re.match(r"03-01-(03|04|05|06)-\d{2}-\d{2}-\d{2}-\d{2}\.wav$", name, re.I)
        if not m:
            return None, None
        label = {"03": "happy", "04": "sad", "05": "angry", "06": "fear"}[m.group(1)]
        actor = re.search(r"Actor_(\d+)", rel, re.I)
        if not actor:
            return None, None
        return label, f"RAVDESS_Actor_{int(actor.group(1)):02d}"
    if source == "SAVEE":
        m = re.match(r"([A-Za-z]{2})_(sa|[afhs])\d+\.wav$", name, re.I)
        if not m:
            return None, None
        label = {"a": "angry", "f": "fear", "h": "happy", "sa": "sad"}[m.group(2).lower()]
        return label, "SAVEE_" + m.group(1).upper()
    if source == "TESS":
        m = re.match(r"([A-Za-z]{3})_.+_(angry|fear|happy|sad)\.wav$", name, re.I)
        if not m:
            return None, None
        return m.group(2).lower(), "TESS_" + m.group(1).upper()
    raise ValueError(f"Beklenmeyen kaynak: {source}")


def read_used_data() -> tuple[set[str], set[str], set[str]]:
    raw = pd.read_csv(ROOT / "ham_veri_listesi.csv")
    used_paths = {normalize_rel(p) for p in raw["path"]}
    used_hashes: set[str] = set()
    missing: list[str] = []
    for rel in sorted(used_paths):
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        used_hashes.add(sha256_file(p))
    if missing:
        raise RuntimeError(f"Egitim manifestindeki {len(missing)} WAV okunamadi: {missing[:5]}")

    used_path_hashes = {path_hash16(rel) for rel in used_paths}

    dedup = pd.read_csv(ROOT / "manifest_v2_dedup_draft.csv", usecols=["path", "sha256", "source_id"])
    used_paths.update(normalize_rel(p) for p in dedup["path"].dropna())
    used_hashes.update(str(h).lower() for h in dedup["sha256"].dropna() if str(h) != "nan")
    used_path_hashes.update(str(s) for s in dedup["source_id"].dropna() if str(s) != "nan")

    local = pd.read_csv(ROOT / "manifest_v2_local.csv", usecols=["path", "source_id"])
    used_paths.update(normalize_rel(p) for p in local["path"].dropna())
    used_path_hashes.update(str(s) for s in local["source_id"].dropna() if str(s) != "nan")

    for fname in ["A1_2/final_oznitelik_seti_A1_2.csv", "final_oznitelik_seti_v2_dedup.csv"]:
        with (ROOT / fname).open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sid = str(row["source_id"])
                used_path_hashes.add(sid)
                used_path_hashes.add(sid.split("__", 1)[0])
    return used_paths, used_hashes, used_path_hashes


def discover_candidates(used_paths: set[str], used_hashes: set[str],
                        used_path_hashes: set[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats: dict = {}
    for source, folder in SOURCE_FOLDERS.items():
        files = sorted((ROOT / folder).rglob("*.wav"))
        labelable: list[dict] = []
        n_target_label = 0
        ignored = 0
        n_used_path = n_used_hash = n_used_path_hash = 0
        for p in files:
            rel = normalize_rel(p.relative_to(ROOT).as_posix())
            label, speaker = parse_label_and_speaker(source, rel)
            if label is None or speaker is None:
                ignored += 1
                continue
            n_target_label += 1
            digest = sha256_file(p)
            phash = path_hash16(rel)
            reasons: list[str] = []
            if rel in used_paths:
                reasons.append("EGITIM_YOLU")
                n_used_path += 1
            if digest in used_hashes:
                reasons.append("EGITIM_ICERIK_SHA256")
                n_used_hash += 1
            if phash in used_path_hashes:
                reasons.append("EGITIM_YOL_HASH")
                n_used_path_hash += 1
            try:
                info = wav_info(p)
            except Exception as exc:
                reasons.append("WAV_OKUNAMADI:" + type(exc).__name__)
                info = {"sample_rate": None, "channels": None, "duration_s": None, "sample_width": None}
            rows.append({
                "kaynak": source,
                "egitim_kaynak_kodu": TRAIN_SOURCE_CODE[source],
                "label": label,
                "konusmaci": speaker.split("_", 1)[1],
                "group_id": speaker,
                "goreli_yol": rel,
                "sha256": digest,
                "yol_hash16": phash,
                "dosya_boyutu": p.stat().st_size,
                **info,
                "dislama_nedeni": "|".join(reasons),
                "secildi": False,
                "secim_sirasi": None,
            })
            if not reasons:
                labelable.append(rows[-1])
        hash_counts = Counter(r["sha256"] for r in labelable)
        for row in labelable:
            if hash_counts[row["sha256"]] > 1:
                row["dislama_nedeni"] = "ADAYLAR_ARASI_ICERIK_TEKRARI"
        stats[source] = {
            "toplam_wav": len(files),
            "hedef_etiketli": n_target_label,
            "diger_duygu_durum": ignored,
            "egitim_yolu": n_used_path,
            "egitim_icerik_hash": n_used_hash,
            "egitim_yol_hash": n_used_path_hash,
            "uygun_aday": sum(1 for r in labelable if not r["dislama_nedeni"]),
        }
    return rows, stats


def rotation_offset(key: str, n: int) -> int:
    return int(sha256_text(key)[:8], 16) % n if n else 0


def select_balanced(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    selection_info: dict = {}
    for source in SOURCE_FOLDERS:
        for cls in CLASSES:
            eligible = [r for r in rows
                        if r["kaynak"] == source and r["label"] == cls and not r["dislama_nedeni"]]
            by_speaker: dict[str, list[dict]] = defaultdict(list)
            for r in eligible:
                by_speaker[r["konusmaci"]].append(r)
            for v in by_speaker.values():
                v.sort(key=lambda r: (r["goreli_yol"], r["sha256"]))

            speakers = sorted(by_speaker)
            chosen: list[dict] = []
            if speakers:
                offset = rotation_offset(f"{source}|{cls}", len(speakers))
                ordered = speakers[offset:] + speakers[:offset]
                cap = max(1, math.ceil(TARGET_PER_SOURCE_CLASS / len(speakers)))
                tally: Counter = Counter()
                for rnd in range(max(len(v) for v in by_speaker.values())):
                    for sp in ordered:
                        if len(chosen) >= TARGET_PER_SOURCE_CLASS:
                            break
                        if tally[sp] >= cap:
                            continue
                        files = by_speaker[sp]
                        if rnd >= len(files):
                            continue
                        chosen.append(files[rnd])
                        tally[sp] += 1
                    if len(chosen) >= TARGET_PER_SOURCE_CLASS:
                        break
                chosen.sort(key=lambda r: (r["konusmaci"], r["goreli_yol"]))

            for i, r in enumerate(chosen, start=1):
                r["secildi"] = True
                r["secim_sirasi"] = i
                r["test_id"] = f"DGT_{TRAIN_SOURCE_CODE[source]}_{cls.upper()}_{i:02d}"
            selected.extend(chosen)
            selection_info[(source, cls)] = {
                "uygun": len(eligible),
                "secilen": len(chosen),
                "eksik": TARGET_PER_SOURCE_CLASS - len(chosen),
                "konusmaci_sayisi": len({r["konusmaci"] for r in chosen}),
                "konusmacilar": ", ".join(sorted({r["konusmaci"] for r in chosen})),
            }
    return selected, selection_info


def write_outputs(rows: list[dict], selected: list[dict], stats: dict,
                  selection_info: dict, used_counts: dict) -> None:
    columns = ["test_id", "kaynak", "egitim_kaynak_kodu", "label", "konusmaci", "group_id",
               "goreli_yol", "sha256", "yol_hash16", "sample_rate", "channels",
               "duration_s", "sample_width", "dosya_boyutu", "secim_sirasi"]
    manifest = pd.DataFrame([
        {k: r.get(k) for k in columns} for r in sorted(selected, key=lambda r: r["test_id"])
    ])
    manifest.to_csv(OUT_DIR / "test_manifest.csv", index=False)

    pool_cols = ["kaynak", "label", "konusmaci", "group_id", "goreli_yol", "sha256",
                 "yol_hash16", "sample_rate", "channels", "duration_s", "dosya_boyutu",
                 "dislama_nedeni", "secildi", "secim_sirasi"]
    pool = pd.DataFrame([{k: r.get(k, "") for k in pool_cols} for r in rows])
    pool["durum"] = pool.apply(
        lambda r: "SECILDI" if r["secildi"] else
                  ("HARIC:" + r["dislama_nedeni"] if r["dislama_nedeni"] else "UYGUN_SECILMEDI"),
        axis=1,
    )
    pool.to_csv(OUT_DIR / "aday_havuzu.csv", index=False)

    pd.DataFrame(LICENSE_ROWS).to_csv(OUT_DIR / "lisans_durumu.csv", index=False)

    protected = []
    for rel in PROTECTED_FILES:
        p = ROOT / rel
        protected.append({"dosya": rel, "sha256": sha256_file(p), "boyut": p.stat().st_size})
    pd.DataFrame(protected).to_csv(OUT_DIR / "korunan_dosya_sha256.csv", index=False)

    lines = [
        "# Dis gercek ses testi: secim, lisans ve cakisma dogrulama raporu",
        "",
        f"Olusturulma (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Bu rapor prediction **oncesinde** uretildi. Secilen kayitlar orijinal WAV "
        "dosyalardir; augmentation, yeniden kodlama veya kesme uygulanmamis, "
        "model tarafinda 22050 Hz / 2.5 s on isleme yapilacaktir.",
        "",
        "## 1. Kapsam ve lisans durumu",
        "",
        "| Kaynak | Durum | Kapsam | Dayanak |",
        "|---|---|---|---|",
    ]
    for r in LICENSE_ROWS:
        lines.append(f"| {r['kaynak']} | {r['durum']} | {r['kapsam']} | {r['kanit']} |")
    lines += [
        "",
        "- CREMA-D ve RAVDESS lisans kosullari erisilebilir birincil kaynaklardan okundu.",
        "- SAVEE icin yetkili lisans/riza metni yerelde ve erisilebilir kurumsal adreslerde "
        "bulunamadi; bu nedenle lisans durumu **DOGRULANAMADI** olarak isaretlendi. "
        "Ucuncu taraf GitHub repolari hak kaynagi kabul edilmedi. Bu test raporu SAVEE "
        "lisansina iliskin yasal bir iddia tasimaz; kayitlar mevcut proje kapsaminda "
        "yerel degerlendirme icin kullanilir, yeniden dagitilmaz.",
        "- BNC (`BNCS-SES-WAV`) 100 kaydi tek konusmaciya (`BNC_SPEAKER_01`) aittir ve "
        "riza/privacy belgesi bulunamadigi icin degerlendirme disinda birakilmistir; "
        "sadece envanterde raporlanir.",
        "- TESS lisansi U of T Borealis metadata'sindan dogrulandi (CC BY-NC 4.0); ancak TESS "
        "mevcut 70/15/15 test bolumunun kaynagidir. 3200 hedef-etiketli TESS WAV'in tamami "
        "ham_veri_listesi.csv'envanterindedir; modeling manifestlerinde yer almayan 1600 WAV "
        "ise icerik SHA-256 bakimindan modeling dosyalariyla birebir aynidir. Bu nedenle TESS "
        "icinde de hash-temiz, kullanilmamis hedef-etiketli aday kalmamistir.",
        "",
        "## 2. Cakisma denetimi",
        "",
        f"- `ham_veri_listesi.csv` benzersiz yol sayisi: {used_counts['paths']}",
        f"- Bu yollardan okunan dosya icerik SHA-256 sayisi: {used_counts['hashes']}",
        f"- Egitimde gorulen benzersiz yol-hash sayisi: {used_counts['path_hashes']}",
        "- Mutlak cakisma kurali: secilen her dosya; (a) egitim manifestindeki yollarla, "
        "(b) egitim dosyalarinin icerik SHA-256 degerleriyle, (c) dedup manifesti ve "
        "A1.2 ozellik CSV'lerindeki source_id/yol hash'leriyle karsilastirildi. "
        "Eslesen adaylar secilmedi.",
        "- Adaylarin kendi aralarinda ayni icerik SHA-256'ya sahip olmasi durumunda "
        "ilgili grup tamamen dislandi.",
        "",
        "| Kaynak | Toplam WAV | Hedef disi | Hedef etiketli | Egitim yolu | Icerik hash | Yol hash | Uygun | Secilen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCE_FOLDERS:
        s = stats[source]
        n_sel = sum(1 for r in selected if r["kaynak"] == source)
        lines.append(
            f"| {source} | {s['toplam_wav']} | {s['diger_duygu_durum']} | {s['hedef_etiketli']} | "
            f"{s['egitim_yolu']} | {s['egitim_icerik_hash']} | {s['egitim_yol_hash']} | "
            f"{s['uygun_aday']} | {n_sel} |"
        )
    lines += [
        "",
        "## 3. Kaynak/sinif dagilimi ve eksikler",
        "",
        "| Kaynak | Sinif | Uygun aday | Secilen | Eksik | Konusmaci sayisi | Konusmacilar |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for (source, cls), info in selection_info.items():
        lines.append(
            f"| {source} | {cls} | {info['uygun']} | {info['secilen']} | {info['eksik']} | "
            f"{info['konusmaci_sayisi']} | {info['konusmacilar']} |"
        )
    shortfalls = [(s, c, i["eksik"]) for (s, c), i in selection_info.items() if i["eksik"] > 0]
    lines += [
        "",
        "Eksik sinif kaydi: " + (
            ", ".join(f"{s}/{c}: -{n}" for s, c, n in shortfalls) if shortfalls else "yok"
        ),
        "",
        "## 4. Otomatik kontroller",
        "",
    ]
    selected_rel = {r["goreli_yol"] for r in selected}
    selected_hash = {r["sha256"] for r in selected}
    checks = {
        "Tum secilen yollar egitim manifestinde yok": not (selected_rel & used_counts["used_paths"]),
        "Tum secilen icerik hash'leri egitim hash'lerinde yok": not (selected_hash & used_counts["used_hashes"]),
        "Tum secilen yol hash'leri egitim source_id'lerinde yok": not (
            {r["yol_hash16"] for r in selected} & used_counts["used_path_hashes"]
        ),
        "Secilenler icinde tekrar eden icerik hash yok": len(selected_hash) == len(selected),
        "Her kaynak/sinif en az 15 kayit": not shortfalls,
        "BNC degerlendirmeye alinmadi": all(r["kaynak"] != "BNC" for r in selected),
        "Tum secilen WAV basliklari okunabildi": all(r.get("sample_rate") for r in selected),
    }
    for name, ok in checks.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
    prediction_ready = all(checks.values())
    lines += [
        "",
        f"**Prediction durumu:** {'HAZIR' if prediction_ready else 'ENGEL'}",
        "" if prediction_ready else
        "Hedef etiketli, ham envanterde yer almayan ve icerik/yol hash acisindan egitimle "
        "cakismayan WAV bulunamadigi icin prediction baslatilmadi. Bu rapor eksiklik "
        "raporudur; basari metrigi uretilmedi.",
        "",
        "## 5. Yorum siniri",
        "",
        "Bu secim rastgele bir referans test kumesi degildir. Egitimde kullanilmamis "
        "orijinal kayitlardan kaynak/sinif dengeli bir **ek degerlendirme** alt kumesidir. "
        "Konusmacilar egitimdeki kayitlarla ayni olabilir; sonuclar gorulmemis konusmaciya "
        "veya genel populyasyona genelleme olarak okunmamalidir.",
    ]
    (OUT_DIR / "cakisma_ve_dogrulama_raporu.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import json
    (OUT_DIR / "secim_durumu.json").write_text(json.dumps({
        "prediction_ready": prediction_ready,
        "failed_checks": [k for k, v in checks.items() if not v],
        "selected_total": len(selected),
        "shortfalls": [
            {"source": s, "class": c, "missing": n}
            for s, c, n in shortfalls
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if not prediction_ready:
        print("UYARI: Prediction engellendi; hash-temiz kullanilmamis hedef-etiketli kayit yok.")
        print("Failed checks:", [k for k, v in checks.items() if not v])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    used_paths, used_hashes, used_path_hashes = read_used_data()
    rows, stats = discover_candidates(used_paths, used_hashes, used_path_hashes)
    selected, selection_info = select_balanced(rows)
    write_outputs(rows, selected, stats, selection_info, {
        "paths": len(used_paths),
        "hashes": len(used_hashes),
        "path_hashes": len(used_path_hashes),
        "used_paths": used_paths,
        "used_hashes": used_hashes,
        "used_path_hashes": used_path_hashes,
    })
    print("Secim tamamlandi:")
    for source in SOURCE_FOLDERS:
        n = sum(1 for r in selected if r["kaynak"] == source)
        print(f"  {source}: {n} kayit")
    print(f"Toplam: {len(selected)} | Cikti: {OUT_DIR}")


if __name__ == "__main__":
    main()
