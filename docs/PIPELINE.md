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

6. Copertura: dalle sole pose si calcola quanto le sagome vincolano la forma
   (vedi "Copertura delle camere" più sotto).

Uscita: `calibration.json` con camere, volume di cattura e un `report` (RMS
complessivo e per camera, frame condivisi per coppia, copertura, versione
OpenCV).

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

Le velocità non partono da zero: ogni frame ha una nuvola di punti propria, e
lo spostamento verso il punto più vicino del frame successivo e precedente
dice come si è mossa la superficie lungo la normale (`estimate_velocities`).
Sui dati sintetici questo riduce di circa un terzo lo scarto tra i punti di un
frame e la superficie del frame dopo nelle parti in moto (95° percentile da 9,6
a 6,8 cm).

**Sincronizzazione sotto il frame (`learn_time_offsets`).** Il modello è
continuo nel tempo, quindi una camera che ha esposto con una frazione di frame
di ritardo può essere renderizzata esattamente a quell'istante. Con questa
opzione il training impara uno scarto temporale per camera (la prima fa da
riferimento) insieme alla scena, e lo riporta in frame e millisecondi. È
pensata per camere senza trigger o genlock; con camere sincronizzate via
hardware va lasciata spenta.

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
| `learn_time_offsets` | false | `true` con camere non sincronizzate via hardware |
| `init_velocities` | true | velocità iniziali stimate dal moto dei punti |
| `static` | false | `true`: 3DGS statico di un solo istante (`static_frame`) |
| `devices` | `auto` | es. `cuda:0` per usare una sola GPU |

#### Riprese lunghe: gli spezzoni

Una ripresa più lunga di una volta e mezza `segment_frames` (60 frame, due
secondi a 30 fps) viene divisa in spezzoni di lunghezza uguale, e ogni
spezzone è un training a sé in `run/seg_000`, `seg_001`, …; `run/run.json` li
elenca. Ogni spezzone vede anche `segment_overlap` frame oltre i propri
estremi, così i frame di confine hanno lo stesso contesto degli altri.

Perché: il modello non ha comunque memoria lunga (una Gaussiana vive pochi
frame), quindi dividere non toglie nulla; in cambio memoria e tempo di ogni
training non dipendono dalla durata, il tetto di Gaussiane (`max_gaussians`)
vale per spezzone e non per tutta la ripresa, e un lavoro interrotto riparte
dal primo spezzone mancante. Rilanciando lo stesso comando gli spezzoni già
fatti con gli stessi dati e le stesse opzioni vengono tenuti (`--fresh` per
rifarli). Gli scarti di tempo tra camere (`learn_time_offsets`) vengono
imparati sul primo spezzone e riusati negli altri.

`eval`, `render`, `export` e `lighten` leggono il risultato attraverso
`run.py` e non distinguono una ripresa intera da una a spezzoni.

#### Scene leggere: Gaussiane che durano più frame

Con `init --frame-step K` la nuvola di partenza viene creata ogni K frame
invece che a ogni frame; le Gaussiane partono con una vita di K frame e la
velocità stimata dal movimento, e descrivono anche i frame intermedi. A parità
di Gaussiane visibili per istante, quelle da memorizzare sono K volte meno.

### `eval`, `render`, `export`, `lighten`, `view`

- `eval`: PSNR, SSIM, IoU della sagoma, PSNR nel riquadro del soggetto e
  sfarfallio, per camera; prima le camere escluse dal training. Lo sfarfallio
  è la variazione tra frame consecutivi che non c'è nelle riprese (in livelli
  a 8 bit): vale zero per un risultato che si muove esattamente come il
  soggetto, qualunque sia il suo errore statico. Per una ripresa a spezzoni lo
  sfarfallio nei frame in cui due spezzoni si incontrano è riportato a parte.
  Scrive `eval/summary.md`, `metrics.json` e immagini verità | render | errore.
- `render`: video del risultato. Di default una camera virtuale percorre
  l'arco coperto dal rig (giro completo solo se le camere circondano il
  soggetto) mentre la ripresa va avanti e indietro; `--take` la riproduce una
  volta dall'inizio alla fine; `--camera` sta dove stava una camera del rig;
  `--path` segue le viste chiave di un file JSON (il player le copia con
  «Copia la vista»). Uscite: `.mp4` (H.264, per guardare), `.mov` (ProRes 422
  HQ, oppure 4444 con canale alfa con `--transparent`) o una cartella di
  immagini PNG; `--width`, `--height` e `--fps` a piacere.
- `export` (default `--format html`): player e scena in un solo file `.html`
  che si apre con doppio clic, senza server né installazione. Per riprese di
  pochi secondi.
- `export --format web`: cartella con il player e la scena a blocchi, caricati
  mentre la ripresa scorre. Si apre con doppio clic su `index.html` (o da un
  server web) e regge riprese di qualunque durata.
- `export --format ply`: un `.ply` 3DGS standard per frame, per altri
  strumenti. È il formato pesante (non compresso).
- `lighten`: copia più leggera di una ripresa già ricostruita, con al massimo
  `--budget` Gaussiane visibili insieme, per dispositivi che ne disegnano
  poche (visori autonomi). Misura quanto ogni Gaussiana contribuisce alle
  immagini delle camere (la somma dei suoi pesi di blending, ricavata con un
  passaggio all'indietro per istante), toglie in ogni istante la stessa quota
  delle meno utili e rifinisce le rimaste con un training breve (15%
  dell'originale) perché coprano il lavoro di quelle tolte.
- `view`: piccolo server HTTP locale (solo `127.0.0.1`) per la cartella `web`.
  Facoltativo: la cartella si apre anche da disco.

## Copertura delle camere

Ogni camera vincola la forma del soggetto solo di traverso alla propria linea
di vista. Viste dall'alto, due camere le cui linee di vista distano `g` gradi
lasciano una forma che, guardata dal centro di quell'angolo, può essere
`1 / cos(g / 2)` volte più larga del vero (un arto tondo diventa un rombo).
Camere una di fronte all'altra contano una volta sola: vedono la stessa sagoma.

| Linee di vista | Angolo massimo | Allargamento nel caso peggiore |
|---|---|---|
| 3 camere a −75°, 25°, 75° (la demo con una camera esclusa) | 100° | +56% |
| 4 camere su un arco di 150° (la demo completa) | 50° | +10% |
| 8 camere ogni 45° (4 linee di vista: le opposte coincidono) | 45° | +8% |
| 9 camere ogni 40° (9 linee di vista) | 20° | +1,5% |

`calibrate` e `train` stampano questo valore per il rig in uso, ed è la
spiegazione principale di ciò che si vede da una camera esclusa: nelle prove
di sviluppo nessuna variante dell'ottimizzazione ha ridotto quell'errore, la
disposizione delle camere sì. Due conseguenze pratiche: conviene distribuire
le linee di vista su 180° senza coppie esattamente opposte (le camere rialzate
e sfalsate della guida servono anche a questo), e una camera tenuta fuori dal
training va scelta sapendo che lascia un buco dove stava.

### Che cosa è stato provato e scartato

Misurato su dati sintetici con geometria nota, senza effetto utile e quindi
non incluso: carving per foto-consistenza del visual hull, scarto casuale di
Gaussiane durante il training, vincolo di coerenza fotometrica tra camere
sulla profondità renderizzata, durate iniziali più brevi, rasterizzazione
antialiasing. Sono prove a bassa risoluzione su CPU: a piena risoluzione su
GPU alcune potrebbero comportarsi diversamente.

## Formato del player

```
player.html                          tutto in un file (riprese brevi)
player/
  index.html, viewer.js, theme.css   player WebGL2 senza dipendenze
  scene.js, scene.json               descrizione: fps, frame, camere, blocchi
  chunks/0000.js, 0001.js, …         un blocco per spezzone della ripresa
```

La scena non è scritta frame per frame. Ogni Gaussiana è salvata **una volta
sola**, con l'istante su cui è centrata, la sua durata e la sua velocità, e il
player la valuta nell'istante mostrato, come fa il training: la sposta di
`velocità × (t − istante)` e ne attenua l'opacità con la campana temporale.

In un blocco le Gaussiane sono ordinate per istante (prima le poche a vita
lunga, che il player considera sempre), 23 byte ciascuna, in cinque parti che
vengono caricate sulla GPU così come sono:

| Parte | Tipo | Contenuto |
|---|---|---|
| A | 4 × uint16 | posizione x y z nel riquadro del blocco, istante |
| B | 4 × uint8 | velocità x y z (con segno; il codice è al quadrato, così i movimenti lenti sono memorizzati finemente), durata |
| C | 4 × uint8 | rotazione (quaternione w x y z) |
| D | 4 × uint8 | logaritmo delle scale x y z, opacità di picco |
| E | 3 × uint8 | colore |

Il blocco è poi compresso con gzip. `scene.json` contiene gli intervalli per
riportare gli interi a metri, frame e così via, e per ogni blocco il numero
massimo di Gaussiane visibili insieme (`alive_max`): è il numero che dice se
un dispositivo riesce a disegnare la scena. I blocchi sono piccoli script
(`mister4dChunk(indice, "base64")`) e non file binari, perché una pagina aperta
da disco può caricare script ma non scaricare dati: così la cartella funziona
con un doppio clic oltre che da un server.

Misure sulla scena della demo (29 frame, 240×135): con il formato precedente,
che riscriveva le Gaussiane vive a ogni frame, il player pesava 3,4 MB; con
questo 1,0 MB; con la scena leggera (`init --frame-step 3`) 0,4 MB.

Il player tiene in memoria il blocco mostrato e il successivo, e mostra la
scena negli istanti su cui è stata allenata (frame interi).

Limiti attuali del player: colore indipendente dalla vista (solo termine DC);
ordinamento in profondità e decodifica dei blocchi sul thread principale (con
blocchi molto grandi è possibile uno scatto al cambio di blocco); nessuna
modalità per visori (WebXR).

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

`tests/test_app.py` copre il server di MISTER 4D Studio: ogni comando che la
pagina può chiedere viene fatto leggere al parser della riga di comando (le
due interfacce non possono divergere senza che un test fallisca), poi coda dei
passi, interruzione, spiegazione degli errori e controlli di accesso (token,
indirizzo locale, file fuori dalla cartella del progetto).
