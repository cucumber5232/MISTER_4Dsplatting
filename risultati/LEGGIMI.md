# Prove su GPU

Questa cartella esiste solo sul ramo `prove-gpu`. Raccoglie riepiloghi, fogli
di immagini e note delle prove fatte sulla workstation con le due RTX 6000
Ada. Il codice provato è quello di `main`: ogni nota riporta il commit.

Qui non ci sono video, checkpoint né dati di training.

## Stato

**Le prove su GPU non sono ancora state fatte.**

Il 24/11/2026 il lavoro è partito su una macchina Linux in cloud, senza GPU e
senza accesso alla workstation. Che cosa è stato controllato da lì, e che cosa
no, è in [senza_gpu/NOTE.md](senza_gpu/NOTE.md).

In questa cartella non ci sono numeri misurati su CPU: non direbbero nulla
sulla qualità a piena risoluzione.

## Come fare le prove sulla workstation

Vanno lanciate sulla workstation, dalla cartella del clone di `main`.

1. Installazione e controllo:

```bat
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

```bat
.venv\Scripts\mister4d doctor
```

2. Secondo worktree per questo ramo, accanto al clone:

```bat
git fetch origin prove-gpu
```

```bat
git worktree add ..\prove-gpu prove-gpu
```

3. Tutte le prove, dalla cartella del clone di `main`:

```bat
.venv\Scripts\python ..\prove-gpu\risultati\prove_gpu.py
```

Lo script crea `bench\D` (960×540, 30 frame) se manca, poi esegue le prove
una dopo l'altra. Per ognuna: training senza le camere `ver01`…`ver05`,
`quality_check.py`, `contact_sheet.py`, `mister4d eval`. Durante il training
legge `nvidia-smi` ogni due secondi. Alla fine copia riepiloghi e immagini in
`risultati\<prova>\` e scrive `NOTE.md`.

Dati e checkpoint restano nel clone (`bench\`, `runs\`, `sessions\`). Lo
script non fa commit e non carica nulla.

Una prova già misurata viene saltata: se il lavoro si interrompe, lo stesso
comando riparte dalla prima prova mancante e ritenta quelle fallite.

Rifare una prova (`--redo`) svuota `runs\<prova>` (per `catena`:
`sessions\persona`) e riscrive la sua cartella qui, compreso `NOTE.md`.

Lo script non svuota una cartella che non ha creato lui, per esempio un
`runs\base` fatto a mano: quella prova si ferma e la nota dice perché. Per
sostituirla serve `--redo`.

4. Guardare le immagini a grandezza piena: `runs\<prova>\evalq\sheet.png` e
   `runs\<prova>\viste.png`. Il PSNR premia la sfocatura quando la geometria è
   incerta. Scrivere il giudizio nella riga «Giudizio a occhio» di ogni
   `NOTE.md`.

5. Caricare i risultati, dal worktree di questo ramo:

```bat
git -C ..\prove-gpu add risultati
```

```bat
git -C ..\prove-gpu commit -m "Add the GPU trials on the benchmark person"
```

```bat
git -C ..\prove-gpu push origin prove-gpu
```

## Le prove

Stesso dataset e stesse camere escluse per tutte. Una modifica alla volta.
Una differenza più piccola di quella tra `base` e `rumore` non è una differenza.

| Prova | Che cosa cambia |
|---|---|
| `base` | niente: opzioni di default, codice così com'è |
| `una_gpu` | come `base`, su una sola GPU (`--devices cuda:0`): dice quanto conta la seconda |
| `rumore` | come `base`, con un altro seme (`seed=1`): dice quanto oscillano le misure da sole |
| `catena` | la catena intera dai video: `mister4d demo --subject human --run --out sessions\persona` |
| `prima` | `lambda_needle=0 min_scale_pixels=0 prune_to_end=false`: com'era prima |
| `decay_0` | `opacity_decay=0` |
| `decay_0015` | `opacity_decay=0.015` (0,005 è il default, quindi è `base`) |
| `entropia_0002`, `entropia_0005`, `entropia_001` | `lambda_opacity_entropy` 0,002, 0,005, 0,01 |
| `spread_0005`, `spread_002` | `lambda_ray_spread` 0,005, 0,02 |
| `dischi` | `init_flatten=0.3 lambda_flat=0.005` |
| `passo3` | `mister4d init --frame-step 3 --exclude ver01 … ver05`, poi training |

`passo3` lavora su una copia del dataset (`bench\D_passo3`), così la nuvola di
partenza delle altre prove non cambia.

## Che cosa c'è in ogni cartella

| File | Da dove viene |
|---|---|
| `NOTE.md` | comandi, tempi, memoria e carico delle GPU, misure, avvisi ed errori dei registri |
| `metrics.json` | `scripts/quality_check.py` (`evalq/metrics.json`) |
| `summary.md`, `eval_metrics.json` | `mister4d eval` (senza le righe per frame) |
| `train_summary.json`, `config.yaml` | il training |
| `gaussiane.json` | opacità e forma delle Gaussiane, lette dal checkpoint |
| `sheet.jpg` | copia di `evalq/sheet.png`: verità, render, errore, alfa sulle camere di verifica |
| `viste.jpg` | copia ridotta del foglio ingrandito di `contact_sheet.py` (`viste.png`) |
| `viste_dettaglio.png` | il centro di due viste di `viste.png`, a grandezza piena e senza compressione |

`RIEPILOGO.md` confronta tutte le prove. `AMBIENTE.md` riporta macchina,
versioni e l'uscita di `mister4d doctor`.

### Perché JPEG

Il foglio ingrandito è un PNG di 3000×3240 pixel. Su una scena di prova
sfocata pesa 7 MB. Con una trama fine, come ci si aspetta da una scena
allenata su GPU, arriva a 16 MB. Con una quindicina di prove il ramo
supererebbe i 100 MB.

Qui vanno quindi copie JPEG con il lato lungo di 1600 pixel, più un ritaglio
PNG a grandezza piena dove aghi e nebbia si vedono senza i segni della
compressione. La stima è meno di un megabyte per prova: va controllata sulle
prime prove vere.

Gli originali restano in `runs\<prova>\` sulla workstation e sono quelli da
guardare per il giudizio. Con `--png` lo script copia gli originali così come
sono.

## Opzioni dello script

| Opzione | Effetto |
|---|---|
| `--list` | elenca le prove |
| `--only base prima` | esegue solo queste |
| `--skip catena` | salta queste |
| `--redo` | rifà anche le prove già presenti |
| `--devices cuda:0` | allena tutte le prove su una sola GPU |
| `--image-side 1600`, `--jpeg-quality 85` | grandezza e qualità delle copie JPEG |
| `--png` | copia i fogli PNG originali, molti megabyte per prova |
| `--smoke` | formato minuscolo e pochi passi: prova lo script, non la qualità |
