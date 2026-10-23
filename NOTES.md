# Status and handoff (Oct 23, 2026)

## Done and delivered (`outputs/`)
Moon (10/18), IC 1318 (x3), M34 re-edit, Double Cluster, NGC 2264, M41, Saturn (video lucky stack),
Sun (10/20 12:11, 774 of 6,450 frames), three shorts with covers, watermark checker.

## In progress: Alhena wide field / Orionid hunt (10/22 03:41–05:35)
- 669 WIDE frames scanned for meteors (difference images + oriented matched filter + Hough, then visual review).
  **Result: no meteors.** Found 3 satellites (frames ~449–451, 501–505, 561–563) and 1 plane (573–574).
  Every 6th frame of the 04:04 session has mount-correction star smears; clouds 04:31–04:55 and after 05:34.
- Sensitivity (injection test): a 0.4 s meteor had to be about magnitude 0 or brighter to be caught in 10 s frames.
- Custom stack of 389 clean frames (jump/cloud frames excluded, hot-pixel + vignetting correction,
  per-frame polynomial sky removal, cubic-polynomial registration ~0.58 px rms, 3-sigma clipped weighted mean)
  -> `work/alhena/alhena_stack2.npy`.
- Plate solved: 26 named stars, 0.49 px rms -> `work/alhena/alh_wcs.json` (crop offset in `alh_crop.npy`).
- Final unsigned image: `masters/alh_v6.png` (1780x996), made by
  `python3 alh_fin2.py alh_v6.png bgt=0.07 k=70 ssat=1.2 glow=0.8 gsig=4 gthr=0.5 ringw=0.65 dsat=0.5`.

## Done Oct 23
- Alhena_Gemini_Orion_OHN.jpg, Alhena_Orionid_Radiant_Labeled_OHN.jpg, OHN_Astro_Short_Orionid_Hunt.mp4 (+ cover) in outputs/.

## Next steps (1–3 done Oct 23)
1. `python3 annotate.py alh_v6.png alh_annot.png` (labels + Orionid radiant at RA 95, Dec +16), check it.
2. Sign both: `SIGDY=58 python3 sign.py alh_v6.png Alhena_Gemini_Orion_OHN.jpg "10.22.26  ·  03:41–05:33" 0.95 "Alhena · Gemini & Orion"`
   (SIGDY lifts the signature so the stamp clears Procyon). Same for the annotated version. Verify with `ohn_watermark_check.py`.
3. Short: join `work/alhena/tl_part*.mp4` -> `met/timelapse.mp4`; make `sat_box.json` for the satellite ring
   (frame 502 trail ≈ window coords cx 396, cy 727); render `short4.py` stills, then video; audio = `audio2.py` with DUR=30.5; mux; captions.
4. Still pending from before: re-stack Saturn from raw FITS and process the 10/19 Moon
   (`DWARF_RAW_TELE_Saturn_EXP_0.0166667_GAIN_40_2026-10-19-22-25-46-246`, `DWARF_RAW_TELE_Moon_EXP_0.0025_GAIN_20_2026-10-19-22-07-28-928` on the drive),
   then rebuild the combined short. Also unprocessed: M34 (10/22, 285 frames) and χ Persei (10/21, 11 frames).

## Rules
- Never post anywhere; provide copy only. "Each platform" = OH HAIL NAH accounts, never Texas Vortex unless named.
- TikTok captions: no links, ≤150 chars + 3–5 hashtags. Shorts titles ≤70 chars; description line 1 ends
  "Watch live 24/7: youtube.com/@ohhailnah/live". Don't state unverified facts (e.g. sunspot region numbers).
- SHO/Hubble palettes are not appropriate for this broadband/duo-band data.

## Oct 23 update
- Re-stacked Alhena with the WIDE camera factory flat (`CALI_FRAME/flat/cam_1`, corners ~0.33 of centre) -> `work/alhena/alhena_stack3_flat.npy`; finals rebuilt from it (`STK=alhena_stack3_flat.npy CT=0.25 python3 alhena.py 140 70`).
- Bias for cam_1 is 240 ADU (matches BLACK=240).
