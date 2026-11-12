# Pipeline: come funziona e che cosa produce

Riferimento tecnico per chi sviluppa. Per l'uso passo passo vedere il README;
per le riprese `SESSIONE_DI_RIPRESA.md`.

```
video per camera ──ingest──► frame allineati ──calibrate──► calibration.json
                                   │                              │
                                   └──────────prepare◄────────────┘
                                                 │
                              dataset (immagini pinhole + meta.json)
                                                 │
                          masks ──► init (visual hull di ogni frame)
                                                 │
                                               train ──► ckpt.pt
                                                 │
                                 eval · render · export ──► player web / .ply
```

Ogni passo legge e scrive cartelle esplicite: si può rifare un passo senza
rifare gli altri, e sostituire un passo con uno strumento esterno purché
rispetti il formato (per esempio maschere fatte con un altro software).

## Convenzioni (uniche per tutto il progetto)

Scritte anche dentro ogni `calibration.json`.

| Grandezza | Convenzione |
|---|---|
| Unità | metri |
| Mondo | destrorso, **+Z verso l'alto** |
| Camera | OpenCV: x a destra, y in basso, z in avanti |
| Estrinseci | mondo → camera: `x_cam = R · x_mondo + t` |
| Intrinseci `K` | pixel OpenCV: il **centro** del pixel in alto a sinistra è (0, 0) |
| Distorsione | OpenCV `(k1, k2, p1, p2, k3)` |
| Quaternioni | `w, x, y, z` |
| Tempo | indice del frame nella clip (anche frazionario) |

Il rasterizzatore campiona il pixel (i, j) in (j + 0,5, i + 0,5): la
conversione di mezzo pixel del punto principale avviene in un solo punto
(`rasterizer.opencv_to_render_K`).

## I passi

### `ingest` — controllo dei file, tempo comune, frame

- `ffprobe` legge risoluzione, frame rate, codec e il timestamp di
  presentazione di ogni frame.
- Ogni frame riceve un indice su una **linea del tempo comune** ricavato dal
  timestamp (non dall'ordine di arrivo) più lo scostamento della camera.
- Scostamenti: correlazione incrociata delle tracce audio (`--sync audio`),
  salto di luminosità (`--sync flash`) o valori manuali. L'errore residuo sotto
  il frame viene riportato, non nascosto.
- Un frame mancante lascia un buco nell'indice: l'istante viene escluso per
  tutte le camere e segnalato.
- Uscita: `frames/<cam>/<indice>.jpg` e `ingest.json` (checksum SHA-256 degli
  originali, scostamenti, frame mancanti, avvisi).

### `calibrate` — intrinseci, estrinseci, sistema mondo

1. Rilevamento ChArUco in ogni frame di ogni camera (OpenCV).
2. Intrinseci e distorsione per camera (`cv2.calibrateCamera`; di default due
   coefficienti radiali, niente tangenziale). Viene riportata la copertura
   dell'immagine da parte del pannello.
3. Posa del pannello per frame (PnP); pose relative tra camere dai frame in
   cui due camere vedono il pannello insieme; concatenazione lungo l'albero
   dei collegamenti più forti. Se il grafo non è connesso il comando si ferma.
4. Bundle adjustment congiunto di pose camera e pose pannello (SciPy,
   Jacobiano sparso, perdita di Huber).
5. Sistema mondo: dal pannello a terra (`--floor-frame`) oppure dal rig
   (origine nel punto più vicino a tutti gli assi ottici; verticale =
   direzione perpendicolare agli assi orizzontali delle immagini, assumendo
   camere montate senza rollio).

Uscita: `calibration.json` con camere, volume di cattura e un `report` (RMS
complessivo e per camera, frame condivisi per coppia, versione OpenCV).

### `prepare` — dataset pronto per il training

- Undistort (una sola volta: il file risultante è marcato `undistorted` e ha
  distorsione nulla) e ridimensionamento con intrinseci aggiornati.
- Con `--empty`: fondale pulito per camera (mediana del set vuoto), trattato
  come i frame.

```
dataset/
  calibration.json     camere pinhole
  meta.json            fps, frame_ids, camere, bbox del soggetto (dopo init)
  images/<cam>/<frame>.jpg
  background/<cam>.png
  masks/<cam>/<frame>.png     255 = soggetto (da `masks` o da altro strumento)
  init_points.ply             punti iniziali di ogni frame (da `init`)
```

### `masks` — soggetto / sfondo

Differenza dal fondale pulito, soglia, soppressione delle ombre (pixel più
scuri ma con la stessa tinta del fondale), pulizia morfologica, componenti
connesse. Scrive `masks_preview.jpg`: **va guardato prima del training**.

### `init` — visual hull, frame per frame

Per **ogni frame**: i voxel (1,5 cm) attorno al soggetto vengono confrontati
con le sagome di tutte le camere e restano quelli che cadono dentro tutte. I
voxel di superficie diventano i punti iniziali di quell'istante, colorati con
la media delle camere. Le camere sono fisse, quindi la proiezione di ogni
voxel si calcola una volta sola e ogni frame costa solo una lettura delle
maschere. Il bounding box del soggetto finisce in `meta.json`.

Usare `--exclude` per le camere che verranno tenute fuori dal training,
altrimenti la valutazione non è indipendente. `--points` fissa il numero
totale di punti (default 600 000, ripartiti sui frame).

### `train` — scena dinamica

Rappresentazione (`model.py`): **Gaussiane locali nel tempo**. Ogni Gaussiana
è una normale Gaussiana 3D (posizione, rotazione, scala, opacità, colore SH)
con tre attributi temporali:

- `times`: l'istante (in frame) attorno a cui esiste;
- `durations`: quanto vive, come deviazione standard (in frame) di una
  campana che ne modula l'opacità;
- `velocities`: un moto lineare (metri per frame) attorno a quell'istante.

Al frame `f` la scena è l'insieme delle Gaussiane vive in `f`, ciascuna
spostata di `velocità × (f − istante)` e attenuata dal proprio peso temporale.
Non c'è nulla da inseguire lungo la clip: un arto veloce è descritto da
Gaussiane che vivono pochi frame, ognuna nata dal visual hull del suo istante;
le parti ferme possono essere descritte da Gaussiane longeve. Tutti i frame
sono ottimizzati insieme e Gaussiane di istanti vicini si sovrappongono nel
tempo, il che attenua gli sfarfallii tipici di una sequenza di 3DGS
indipendenti.

> Durante lo sviluppo è stata provata anche la rappresentazione "Gaussiane
> canoniche + rete di deformazione": con poche camere seguiva bene i moti
> piccoli ma perdeva gli arti in moto rapido (l'ottimizzatore trova più
> economico spegnerli che spostarli). È il motivo della scelta attuale; resta
> una strada per il futuro, con vincoli aggiuntivi (vedi roadmap nel README).

Ottimizzazione (`trainer.py`):

- a ogni passo ogni GPU riceve un istante casuale e alcune camere; perdita
  L1 + D-SSIM sull'immagine composta su uno sfondo casuale, più:
  - sagoma: alpha renderizzata contro la maschera;
  - scala: penalità su Gaussiane troppo grandi o troppo allungate;
  - velocità: piccola penalità che impedisce derive lungo le direzioni che
    nessuna camera osserva;
- densificazione con la strategia di default di gsplat (duplica / dividi /
  elimina) fino al 60% delle iterazioni, con tetto `max_gaussians`;
- numero di passi automatico: 250 per frame della clip (minimo 7000).

**Multi-GPU.** I parametri vivono sulla prima GPU. A ogni passo ciascuna GPU
riceve una copia differenziabile dei parametri, renderizza un batch diverso e
restituisce la propria perdita; un solo `backward` riporta i gradienti di
tutte le GPU al proprietario. Con due GPU ogni passo vede il doppio delle
immagini in un tempo simile. I frame vengono decodificati in thread separati
(`loader_threads`) e tenuti in RAM fino a `cache_gb`.

Opzioni: `configs/default.yaml` e `TrainConfig` in `trainer.py`. Le più utili:

| Opzione | Default | Quando cambiarla |
|---|---|---|
| `iterations` | -1 (automatico) | aumentare se il risultato è ancora sfocato |
| `holdout` | `[]` | camere escluse dal training, usate da `eval` |
| `views_per_step` | 4 | camere per GPU a ogni passo |
| `sh_degree` | 0 | 1–2 solo con molte camere (colore dipendente dalla vista) |
| `max_gaussians` | 2 000 000 | limite di memoria |
| `init_duration` | 1.0 | vita iniziale di una Gaussiana, in frame inizializzati |
| `static` | false | `true`: 3DGS statico di un solo istante (`static_frame`) |
| `devices` | `auto` | es. `cuda:0` per usare una sola GPU |

### `eval`, `render`, `export`, `view`

- `eval`: PSNR, SSIM, IoU della sagoma e PSNR nel riquadro del soggetto, per
  camera; prima le camere escluse dal training. Scrive `eval/summary.md`,
  `metrics.json` e immagini verità | render | errore.
- `render`: video da una camera virtuale che percorre l'arco coperto dal rig
  (giro completo solo se le camere circondano il soggetto).
- `export` (default `--format html`): player e dati in un solo file `.html`
  che si apre con doppio clic, senza server né installazione.
- `export --format web`: cartella con il player e un file compatto per frame
  (vedi sotto), da servire con `view`; più leggera da caricare per clip lunghe.
- `export --format ply`: un `.ply` 3DGS standard per frame, per altri
  strumenti. È il formato pesante (non compresso).
- `view`: piccolo server HTTP locale (solo `127.0.0.1`) per la cartella `web`:
  serve perché i browser non lasciano leggere file separati da `file://`.

## Formato del player web

```
export/
  index.html, viewer.js      player WebGL2 senza dipendenze
  scene.json                 metadati: fps, numero di Gaussiane per frame, bounding box, camere
  frames/<indice>.bin        le Gaussiane vive in quel frame
```

Ogni frame occupa 17 byte per Gaussiana: posizione a 16 bit nel bounding box
(6), quaternione a 8 bit (4), logaritmo delle scale a 8 bit (3), colore e
opacità (4). Gli array sono riempiti fino a `tex_width × tex_height` per
essere caricati direttamente come texture. Ordine di grandezza: 40 000
Gaussiane vive per frame × 150 frame ≈ 100 MB.

Il formato `html` contiene gli stessi byte, codificati in base64 dentro la
pagina (circa un terzo in più di spazio), insieme a `scene.json` e al codice
del player.

Limiti attuali del player: colore indipendente dalla vista (solo termine DC),
tutti i frame tenuti in memoria, ordinamento in profondità sul thread
principale.

## Backend di rasterizzazione

`rasterizer.py` espone una sola funzione `render` con due backend:

- `gsplat` (GPU): quello da usare.
- `torch` (CPU): riferimento lento scritto per questo progetto, con la stessa
  matematica; serve a provare tutta la pipeline senza GPU (test, demo).

`mister4d doctor` renderizza la stessa scena con entrambi su ogni GPU e
confronta le immagini: è il collaudo del backend GPU sulla macchina reale.

## Test

```bash
pytest -m "not slow"
```

```bash
pytest
```

I test lenti allenano un modello minuscolo su CPU (anche nel percorso
multi-device) e calibrano un rig sintetico confrontandolo con la verità.
