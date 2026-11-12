# MISTER 4D Splatting

Proof of concept per ricostruire una persona in **4D** (Gaussian Splatting
dinamico) a partire dalle registrazioni di **3 o più camere fisse**, e
rivederla da nuovi punti di vista.

Segue l'impostazione della *Guida operativa set 4DGS* di MISTER Smart
Innovation: acquisizione sincronizzata, ricostruzione offline, riproduzione.
Questo repository copre la parte software dalla ricezione dei file video al
player.

```
video per camera → ingest → calibrate → prepare → masks → init → train → eval / render / export → view
```

| Passo | Che cosa fa |
|---|---|
| `ingest` | controlla i file, allinea le camere nel tempo (audio, flash o manuale), segnala i frame mancanti, estrae i frame |
| `calibrate` | intrinseci, distorsione ed estrinseci del rig da un pannello ChArUco; sistema mondo metrico |
| `prepare` | undistort e ridimensionamento con intrinseci coerenti; fondale del set vuoto |
| `masks` | maschere soggetto/sfondo per confronto con il set vuoto |
| `init` | visual hull di ogni frame: geometria iniziale e ingombro del soggetto |
| `train` | ottimizzazione della scena dinamica su tutte le GPU disponibili |
| `eval` | metriche e immagini di confronto su una camera esclusa dal training |
| `render` | video da una camera virtuale che gira attorno al soggetto |
| `export` | player 4D in **un solo file HTML** che si apre con doppio clic; in alternativa cartella per `view` o sequenza di `.ply` |

Tutto gira **in locale** sulla propria macchina: non c'è nessun servizio web o
cloud, e nessun dato lascia il computer. Il player è una pagina HTML letta dal
browser direttamente da disco.

## Licenze: perché si può usare in azienda

Il codice è **scritto da zero** e usa solo componenti con licenze permissive:
[gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0) per la
rasterizzazione GPU, PyTorch, OpenCV, NumPy, SciPy. Non contiene né richiede
il codice Inria di 3D Gaussian Splatting (licenza di sola ricerca), né i
progetti che ne dipendono (4DGaussians, Dynamic3DGaussians), né modelli
corporei a licenza commerciale (SMPL). Dettagli, esclusioni e punti ancora da
decidere — tra cui la **licenza di questo stesso repository, non ancora
scelta** — in [docs/LICENZE.md](docs/LICENZE.md).

## Stato del progetto

Verificato in questo repository, su dati sintetici con verità nota
(`pytest`) e con la demo completa a partire dai video:

- sincronizzazione: offset di avvio delle camere ritrovati esattamente, sia
  dall'audio sia dal flash; un frame perso viene segnalato e non rinumerato;
- calibrazione ChArUco: errore di riproiezione 0,12 px; posizioni delle camere
  entro 5 mm e focali entro lo 0,2% dai valori veri;
- ricostruzione dinamica: vedi [Risultato della demo](#risultato-della-demo).

**Non ancora verificato** (lo sviluppo è avvenuto su una macchina Linux senza
GPU e senza riprese reali):

- lo script di installazione per Windows: le versioni che installa esistono
  sugli indici ufficiali di PyTorch e gsplat, ma lo script non è stato
  eseguito su una macchina Windows;
- il backend GPU (gsplat/CUDA) e il training su più GPU su hardware vero. La
  logica multi-GPU è collaudata su CPU con due dispositivi virtuali e il
  rasterizzatore di riferimento su CPU coincide numericamente con
  l'implementazione di riferimento di gsplat; il primo comando da lanciare
  sulla workstation è `mister4d doctor`, che confronta i due backend su ogni
  GPU;
- qualità e tempi su riprese reali: i valori di default sono punti di
  partenza ragionevoli, non parametri tarati.

## Installazione

Servono FFmpeg e, per il training, una GPU NVIDIA con driver aggiornato.

### Windows

Prerequisiti: **Python 3.10** (l'installer più recente su python.org è il
3.10.11; in alternativa `winget install Python.Python.3.10`), Git e FFmpeg
(`winget install Gyan.FFmpeg`). Non servono Visual Studio né CUDA Toolkit.

```bat
git clone https://github.com/cucumber5232/MISTER_4Dsplatting.git
```

```bat
cd MISTER_4Dsplatting
```

Doppio clic su `scripts\install.bat`, oppure da terminale:

```bat
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

Lo script crea l'ambiente virtuale `.venv`, installa PyTorch 2.4.1 per CUDA
12.4, la wheel **precompilata** di gsplat 1.5.3 e il progetto, poi lancia
`mister4d doctor`. La versione di Python è vincolata: le wheel ufficiali
precompilate di gsplat per Windows esistono solo per Python 3.10 con PyTorch
2.4 e CUDA 12.4. Con altre versioni gsplat va compilato (Visual Studio Build
Tools e CUDA Toolkit, vedi `docs/INSTALL_WIN.md` nel repository di gsplat).

In ogni nuovo terminale attivare l'ambiente dalla cartella del progetto:

```bat
.venv\Scripts\activate.bat
```

### Linux

Python ≥ 3.10. Lo script crea `.venv`, installa tutto e lancia il controllo
della macchina:

```bash
git clone https://github.com/cucumber5232/MISTER_4Dsplatting.git
```

```bash
cd MISTER_4Dsplatting && scripts/install.sh
```

```bash
source .venv/bin/activate
```

### Controllo

Dopo l'installazione il programma è il comando `mister4d`, disponibile quando
l'ambiente virtuale è attivo. Controllo dell'ambiente e collaudo delle GPU:

```bash
mister4d doctor
```

## Prova senza camere (dati sintetici)

Genera una sessione di ripresa sintetica (manichino animato, 4 camere con
distorsione, avvii sfalsati, traccia audio con battito di mani, pannello
ChArUco, set vuoto) ed esegue tutta la pipeline, dai video al player:

```bash
mister4d demo --out sessions/demo --run
```

Con GPU lavora a piena risoluzione con le opzioni di default; senza GPU
ripiega su immagini 240×135 e sul rasterizzatore di riferimento su CPU (circa
un quarto d'ora). Al termine aprire con doppio clic
`sessions/demo/player.html`.

### Risultato della demo

Demo eseguita su CPU a 240×135 px, 29 frame, training su 3 camere
(`cam01`, `cam03`, `cam04`) e `cam02` esclusa e usata come verifica:

| | PSNR soggetto | IoU sagoma |
|---|---|---|
| camera esclusa (nuovo punto di vista) | 21,1 dB | 0,87 |
| camere di training | 35,1 dB | 0,99 |

Sono numeri di una prova minuscola su CPU: servono a dire che la catena
funziona da un capo all'altro, non a prevedere la qualità su riprese reali.

## Flusso con riprese reali

Che cosa registrare e come: [docs/SESSIONE_DI_RIPRESA.md](docs/SESSIONE_DI_RIPRESA.md).
In breve servono tre clip, ciascuna con un video per camera e gli stessi nomi
di file (`cam01.mp4`, `cam02.mp4`, …): pannello ChArUco, set vuoto, soggetto.

```bash
mister4d board --out charuco_5x7.png
```

```bash
mister4d ingest raw/calib --out work/calib --sync audio --start 30
```

```bash
mister4d ingest raw/empty --out work/empty
```

```bash
mister4d ingest raw/take01 --out work/take01 --sync audio --start 30
```

```bash
mister4d calibrate work/calib --out work/calibration.json --square 0.12 --marker 0.09
```

```bash
mister4d prepare work/take01 --calibration work/calibration.json --empty work/empty --out data/take01 --downscale 2
```

```bash
mister4d masks data/take01
```

Guardare `data/take01/masks_preview.jpg` prima di proseguire.

```bash
mister4d init data/take01 --exclude cam02
```

```bash
mister4d train data/take01 --out runs/take01 --config configs/default.yaml --holdout cam02
```

```bash
mister4d eval runs/take01
```

```bash
mister4d render runs/take01 --out runs/take01/orbit.mp4
```

```bash
mister4d export runs/take01 --out exports/take01.html
```

`exports/take01.html` contiene player e dati: si apre con doppio clic e si può
consegnare così com'è.

Note:

- I comandi sono identici su Windows e Linux (i percorsi con `/` funzionano
  anche su Windows). A `ingest` si passa la cartella della clip: prende tutti
  i video che contiene.
- `--sync`: `none` per camere con trigger/genlock avviate insieme, `audio`
  (battito di mani), `flash`, oppure `--offsets cam02=3,…`. `--start N` scarta
  i primi N frame dopo l'allineamento.
- `--square` / `--marker`: dimensioni **misurate** sul pannello stampato.
- `--holdout` / `--exclude`: la camera indicata resta fuori da inizializzazione
  e training e serve a misurare la qualità da un punto di vista indipendente.
  Per il risultato finale si rilancia senza, usando tutte le camere.
- `export --format web` scrive invece una cartella (un file per frame) da
  aprire con `mister4d view <cartella>`: più leggera da caricare per clip
  lunghe. `--format ply` scrive un `.ply` 3DGS per frame.
- Ogni comando ha `--help`; tutte le opzioni di training sono in
  `configs/default.yaml` e si cambiano con `--set chiave=valore`.

## Training sulle GPU

`train` usa di default **tutte le GPU visibili** (`devices: auto`). A ogni
passo ogni GPU renderizza un istante diverso della clip visto da
`views_per_step` camere; un solo passo di ottimizzazione raccoglie i gradienti
di tutte. Con due GPU ogni passo vede quindi il doppio delle immagini in un
tempo simile. Per limitarsi a una GPU:

```bash
mister4d train data/take01 --out runs/take01 --devices cuda:0
```

I frame vengono decodificati da thread in background e tenuti in RAM
(`cache_gb`, default 16 GB): se la clip preparata entra in memoria conviene
alzarlo; altrimenti ridurre la risoluzione con `prepare --downscale`.

Il numero di passi è automatico (250 per frame della clip, minimo 7000): ogni
frame ha le proprie Gaussiane, quindi clip più lunghe richiedono più passi.

## Come è rappresentato il 4D

Gaussiane **locali nel tempo**: ogni Gaussiana ha, oltre agli attributi 3D,
un istante, una durata e una velocità. Nasce dal visual hull del proprio frame
e vive pochi frame, sovrapponendosi a quelle degli istanti vicini. Non serve
inseguire le parti del corpo lungo la clip, quindi un gesto rapido non viene
perso; tutti i frame sono ottimizzati insieme. Dettagli e motivazione della
scelta in [docs/PIPELINE.md](docs/PIPELINE.md).

## Limiti noti

- **Poche camere = un settore.** Con 3–4 camere la ricostruzione è credibile
  dai punti di vista vicini a quelli ripresi; il retro e le zone mai viste non
  vengono inventate. `render` percorre solo l'arco coperto dal rig.
- **Sincronizzazione.** Il software allinea al frame più vicino e riporta
  l'errore residuo; non può correggere esposizioni non simultanee. Per gesti
  rapidi serve sincronizzazione hardware.
- **Maschere.** Dipendono dal set vuoto: luce o inquadratura che cambiano tra
  le due riprese le rovinano; le ombre nette a terra possono entrare nella
  maschera. Si possono fornire maschere fatte con altri strumenti.
- **Coerenza temporale.** Le Gaussiane non sono le stesse da un istante
  all'altro: da punti di vista lontani dalle camere possono restare piccoli
  sfarfallii, e non c'è una traiettoria per punto utilizzabile per analisi del
  gesto.
- **Player.** Colore indipendente dalla vista, tutti i frame in memoria: va
  bene per clip di qualche secondo, non per riprese lunghe.
- **Nessuna misura.** Fotorealismo non significa accuratezza metrica: il
  risultato non è una superficie su cui prendere quote.

## Struttura

```
src/mister4d/     pacchetto Python (un modulo per passo della pipeline)
  viewer/         player WebGL2
configs/          opzioni di training
scripts/          installazione (Windows e Linux)
tests/            test automatici
docs/             licenze, pipeline, sessione di ripresa
```

Test:

```bash
pytest
```

## Roadmap

Riferita alle fasi della guida operativa.

- **B. Metodo** — provare il backend GPU sulla workstation (`doctor`,
  `demo --run` a piena risoluzione) e misurare tempi e memoria con una e due GPU.
- **C–D. Due camere e calibrazione** — prima ripresa reale: pannello, set
  vuoto, oggetto fermo; verificare `ingest`, sync e RMS di calibrazione;
  aggiungere il controllo su punti 3D indipendenti.
- **E. Quattro viste** — breve azione reale; tarare soglie delle maschere,
  densificazione e numero di passi; valutare maschere da modello di
  segmentazione con licenza compatibile.
- **F–G. Rig esteso e qualità** — con 8+ camere: `sh_degree` > 0, camera di
  verifica dedicata, regolarizzazione temporale (durate più lunghe per le
  parti ferme), eventuale ritorno a un modello a deformazione per avere
  traiettorie per punto.
- **H. Player** — streaming dei frame, compressione, colore dipendente dalla
  vista, integrazione con il motore XR di destinazione.
- **Acquisizione** — registratore con controllo camere (API Z CAM / Blackmagic
  / PTZOptics): fuori dallo scopo di questo POC, che parte dai file.
