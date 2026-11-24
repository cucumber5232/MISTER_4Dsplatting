# Pipeline

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

Ogni passo legge e scrive cartelle sue. Si può rifare un passo senza rifare
gli altri, oppure sostituirlo con uno strumento esterno che rispetti il
formato, per esempio maschere fatte con un altro software.

## Convenzioni

Scritte anche dentro ogni `calibration.json`.

| Grandezza | Convenzione |
|---|---|
| Unità | metri |
| Mondo | destrorso, +Z verso l'alto |
| Camera | OpenCV: x a destra, y in basso, z in avanti |
| Estrinseci | mondo → camera: `x_cam = R · x_mondo + t` |
| Intrinseci `K` | pixel OpenCV: il centro del pixel in alto a sinistra è (0, 0) |
| Distorsione | OpenCV `(k1, k2, p1, p2, k3)` |
| Quaternioni | `w, x, y, z` |
| Tempo | indice del frame nella clip (anche frazionario) |

Il rasterizzatore campiona il pixel (i, j) in (j + 0,5, i + 0,5): la
conversione di mezzo pixel del punto principale avviene in un solo punto
(`rasterizer.opencv_to_render_K`).

## I passi

### `ingest`: controllo dei file, tempo comune, frame

- `ffprobe` legge risoluzione, frame rate, codec e il timestamp di
  presentazione di ogni frame.
- Ogni frame riceve un indice su una linea del tempo comune, ricavato dal
  timestamp più lo scostamento della camera.
- Scostamenti: correlazione incrociata delle tracce audio (`--sync audio`),
  salto di luminosità (`--sync flash`) o valori manuali. L'errore residuo sotto
  il frame viene riportato.
- Un frame mancante lascia un buco nell'indice. L'istante viene escluso per
  tutte le camere e segnalato.
- Uscita: `frames/<cam>/<indice>.jpg` e `ingest.json` (checksum SHA-256 degli
  originali, scostamenti, frame mancanti, avvisi).

### `calibrate`: intrinseci, estrinseci, sistema mondo

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

### `prepare`: dataset pronto per il training

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

### `masks`: soggetto / sfondo

Differenza dal fondale pulito, soglia, soppressione delle ombre (pixel più
scuri ma con la stessa tinta del fondale), pulizia morfologica, componenti
connesse. Scrive `masks_preview.jpg`, da guardare prima del training.

### `init`: visual hull, frame per frame

Per ogni frame i voxel (1,5 cm) attorno al soggetto vengono confrontati
con le sagome di tutte le camere e restano quelli che cadono dentro tutte. I
voxel di superficie diventano i punti iniziali di quell'istante, colorati con
la media delle camere. Le camere sono fisse, quindi la proiezione di ogni
voxel si calcola una volta sola e ogni frame costa solo una lettura delle
maschere. Il bounding box del soggetto finisce in `meta.json`.

Usare `--exclude` per le camere che verranno tenute fuori dal training,
altrimenti la valutazione non è indipendente. `--points` fissa il numero
totale di punti (default 600 000, ripartiti sui frame).

### `train`: scena dinamica

Rappresentazione (`model.py`): Gaussiane locali nel tempo. Ogni Gaussiana
è una normale Gaussiana 3D (posizione, rotazione, scala, opacità, colore SH)
con tre attributi temporali:

- `times`: l'istante (in frame) attorno a cui esiste;
- `durations`: quanto vive, come deviazione standard (in frame) di una
  campana che ne modula l'opacità;
- `velocities`: un moto lineare (metri per frame) attorno a quell'istante.

Al frame `f` la scena è l'insieme delle Gaussiane vive in `f`, ciascuna
spostata di `velocità × (f − istante)` e attenuata dal proprio peso temporale.
Un arto veloce è descritto da Gaussiane che vivono pochi frame, ognuna nata
dal visual hull del suo istante. Le parti ferme possono essere descritte da
Gaussiane che durano di più. Tutti i frame sono ottimizzati insieme e le
Gaussiane di istanti vicini si sovrappongono nel tempo, cosa che riduce lo
sfarfallio.

È stata provata anche la rappresentazione con Gaussiane canoniche e una rete
di deformazione. Con poche camere seguiva bene i moti piccoli ma perdeva gli
arti in moto rapido, perché all'ottimizzatore costa meno spegnerli che
spostarli.

Le velocità non partono da zero. Ogni frame ha la sua nuvola di punti, e lo
spostamento verso il punto più vicino del frame successivo e precedente dice
come si è mossa la superficie lungo la normale (`estimate_velocities`).
Sui dati sintetici questo riduce di circa un terzo lo scarto tra i punti di un
frame e la superficie del frame dopo nelle parti in moto (95° percentile da 9,6
a 6,8 cm).

Sincronizzazione sotto il frame (`learn_time_offsets`). Il modello è continuo
nel tempo, quindi una camera che ha esposto con una frazione di frame di
ritardo può essere disegnata a quell'istante. Con questa opzione il training
impara uno scarto di tempo per camera insieme alla scena (la prima camera fa
da riferimento) e lo riporta in frame e millisecondi. Serve per camere senza
trigger o genlock. Con camere sincronizzate via hardware va lasciata spenta.

Colori delle camere (`learn_camera_colors`). Camere di modello diverso, o con
esposizione e bianco automatici, rendono lo stesso soggetto con colori
diversi, e la scena cambia tinta secondo il punto di vista. Con questa
opzione il training impara per ogni camera un guadagno per rosso, verde e
blu. I guadagni hanno media nulla tra le camere, quindi la scena prende
l'aspetto medio. Vengono imparati dal primo passo, quando i colori di
partenza sono ancora la media delle camere. In una ripresa a spezzoni vengono
imparati sul primo e riusati negli altri. `eval` confronta ogni camera usata
con la scena vista attraverso il suo guadagno. Con camere uguali e
impostazioni bloccate va lasciata spenta.

Ottimizzazione (`trainer.py`):

- a ogni passo ogni GPU riceve un istante casuale e alcune camere; perdita
  L1 + D-SSIM sull'immagine composta su uno sfondo casuale, più:
  - sagoma: alpha renderizzata contro la maschera;
  - scala: penalità su Gaussiane troppo grandi o troppo allungate;
  - velocità: piccola penalità che impedisce derive lungo le direzioni che
    nessuna camera osserva;
- decadimento dell'opacità (`opacity_decay`): dopo ogni passo le Gaussiane
  disegnate in quel passo perdono lo 0,5% dell'opacità. Quelle che servono
  alle immagini lo riprendono subito dal gradiente. Le altre si spengono e
  vengono eliminate dalla densificazione. Non agisce quando si rifinisce una
  scena già fatta (`lighten`). Le misure sono in
  [Idee recenti](#idee-recenti);
- forma delle Gaussiane: una penalità su quelle ad ago, cioè molto più lunghe
  in una direzione che nelle altre due (`lambda_needle`), e una dimensione
  minima pari a mezzo pixel delle immagini di training (`min_scale_pixels`).
  Un ago nasce quando una Gaussiana si allunga lungo la linea di vista di una
  camera: da lì non si vede, da ogni altro punto è una striscia. Una Gaussiana
  più sottile di un pixel non costa nulla nelle immagini di training e
  ingrandita diventa un pelo. I dischi appoggiati su una superficie restano
  permessi;
- densificazione con la strategia di default di gsplat (duplica / dividi /
  elimina) fino al 60% delle iterazioni, con tetto `max_gaussians`;
- dopo la fine della densificazione le Gaussiane che scendono sotto la soglia
  di opacità continuano a essere tolte, fino all'ultimo passo (`prune_to_end`);
- numero di passi automatico: 250 per frame della clip (minimo 7000).

Multi-GPU. I parametri stanno sulla prima GPU. A ogni passo ciascuna GPU
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
| `learn_camera_colors` | false | `true` con camere che non rendono i colori allo stesso modo |
| `lambda_needle` | 0.02 | 0: nessun limite alla forma ad ago |
| `min_scale_pixels` | 0.5 | 0: nessuna dimensione minima; valori più alti danno una scena più morbida |
| `opacity_decay` | 0.005 | 0.015: meno Gaussiane e punti di vista intermedi più stabili, ma camere usate meno nitide; 0: spento |
| `init_velocities` | true | velocità iniziali stimate dal moto dei punti |
| `static` | false | `true`: 3DGS statico di un solo istante (`static_frame`) |
| `devices` | `auto` | es. `cuda:0` per usare una sola GPU |

#### Riprese lunghe: gli spezzoni

Una ripresa più lunga di una volta e mezza `segment_frames` (60 frame, due
secondi a 30 fps) viene divisa in spezzoni di lunghezza uguale, e ogni
spezzone è un training a sé in `run/seg_000`, `seg_001`, …; `run/run.json` li
elenca. Ogni spezzone vede anche `segment_overlap` frame oltre i propri
estremi, così i frame di confine hanno lo stesso contesto degli altri.

Una Gaussiana vive pochi frame, quindi dividere la ripresa non toglie nulla.
Memoria e tempo di ogni training non dipendono più dalla durata, il tetto di
Gaussiane (`max_gaussians`) vale per spezzone, e un lavoro interrotto riparte
dal primo spezzone mancante. Rilanciando lo stesso comando gli spezzoni già
fatti con gli stessi dati e le stesse opzioni vengono tenuti (`--fresh` per
rifarli). Gli scarti di tempo tra camere (`learn_time_offsets`) e i loro
colori (`learn_camera_colors`) vengono imparati sul primo spezzone e riusati
negli altri.

`eval`, `render`, `export` e `lighten` leggono il risultato attraverso
`run.py` e non distinguono una ripresa intera da una a spezzoni.

#### Scena intera: il set e la ripresa

Con `init --full-scene` e `train --full-scene` la ripresa non viene ritagliata
dal fondo. Il set non si muove, quindi viene ricostruito una volta sola e poi
resta fisso sotto tutti gli istanti (`scenery.py`, e `scenery` in `model.py`).

1. Immagini del set vuoto, una per camera (`background/`). Si usano quelle
   registrate con `prepare --empty`. Se mancano vengono stimate dalla ripresa
   come mediana nel tempo di ogni pixel.
2. Punti di partenza del set (`scenery/init_points.ply`), senza modelli
   esterni. Per ogni camera si fa scorrere un piano nella scena, si riportano
   su di esso le immagini delle due camere più vicine e ogni pixel tiene il
   piano su cui le immagini concordano meglio (correlazione normalizzata su
   finestre 7×7). I piani sono di due famiglie: di fronte alla camera, 768 a
   passo costante in profondità inversa, per pareti e fondali, e orizzontali,
   uno ogni centimetro, per i pavimenti. Una profondità vale solo se un'altra
   camera trova una superficie nello stesso punto (entro il 3%). Dove le
   camere non concordano, per esempio su una parete uniforme, il punto prende
   una media morbida delle profondità vicine.
3. Punti di partenza di ciò che si muove: il visual hull, calcolato su tutto
   ciò che differisce dal set vuoto, ombre comprese (`change/`). Queste
   sagome servono solo a mettere le prime Gaussiane. Il training non usa
   maschere.
4. Training del set come scena statica sulle immagini del set vuoto, in
   `run/scenery/`. Opzioni: `scenery_iterations` (default 7000),
   `scenery_max_gaussians`, `scenery_max_scale`, `scenery_prune_scale`.
5. Training della ripresa sulle immagini intere. Le Gaussiane del set sono
   congelate e vengono disegnate insieme a quelle della ripresa, ma non
   vengono più ottimizzate né densificate. Quello che cambia rispetto al set,
   ombra del soggetto compresa, è descritto dalle Gaussiane della ripresa.

Una ripresa lunga ha un solo set per tutti gli spezzoni. Nei file per il
player le Gaussiane del set stanno all'inizio di ogni blocco con durata 255
(sempre visibili), quindi player e lettore Unity le disegnano senza
modifiche. In una ripresa a spezzoni il set viene ripetuto in ogni blocco.
`lighten` lascia il set com'è e alleggerisce solo ciò che si muove.

Prova su CPU (persona di prova, 4 camere, 240×135, 250 passi per fase): i
punti di partenza del set, visti da tre camere di verifica, coprono dal 92 al
99% dell'immagine con un errore mediano di 4–7 livelli su 255. Il set
addestrato dà 22 dB sulle camere di verifica. Non è mai stato addestrato su
GPU a piena risoluzione.

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
- `export --format engine`: cartella per un motore grafico. Gli stessi blocchi
  del player come file binari (`chunks/0000.bin`, gzip) e uno `scene.json`
  con soli numeri, testi ed elenchi di numeri. La legge il componente per
  Unity in `unity/`; il formato è descritto in `unity/README.md`.
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
spiegazione principale di ciò che si vede da una camera esclusa. Nelle prove
nessuna variante dell'ottimizzazione ha ridotto quell'errore, mentre cambiare
la disposizione delle camere lo riduce. Conviene quindi distribuire le linee
di vista su 180° senza coppie esattamente opposte, come fanno le camere
rialzate e sfalsate della guida. Una camera tenuta fuori dal training lascia
un buco dove stava.

### Che cosa è stato provato e scartato

Misurato su dati sintetici con geometria nota, senza effetto utile e quindi
non incluso: carving per foto-consistenza del visual hull, scarto casuale di
Gaussiane durante il training, vincolo di coerenza fotometrica tra camere
sulla profondità renderizzata, durate iniziali più brevi, rasterizzazione
antialiasing. Sono prove a bassa risoluzione su CPU: a piena risoluzione su
GPU alcune potrebbero comportarsi diversamente.

## Qualità

Tre strumenti in `scripts/`, da lanciare con il Python di `.venv`, servono a
giudicare quanto una scena sembra vera.

- `make_benchmark.py` crea riprese sintetiche della persona di prova con la
  verità: per ogni camera e frame la profondità e la sagoma esatte. Oltre alle
  camere di training (`cam01`…) ci sono camere di verifica (`ver01`…) messe
  negli spazi tra le prime, oltre un'estremità dell'arco e più in alto. Non
  vanno mai usate per `init` e `train`.
- `quality_check.py` misura una scena ricostruita, separando camere di
  training e di verifica.
- `contact_sheet.py` disegna la scena da punti di vista lungo e oltre l'arco
  delle camere, anche ingrandita.

Misure di `quality_check.py`, oltre a PSNR, SSIM, IoU e sfarfallio:

| Misura | Che cosa dice |
|---|---|
| `spill` | alfa disegnata fuori dalla sagoma vera, in rapporto alla sua area |
| `holes` | alfa mancante dentro la sagoma vera |
| `soft` | quota dei pixel attorno al soggetto con alfa né vuota né piena |
| `depth` | distanza mediana tra profondità disegnata e vera |
| `thick` | quanto è spesso in profondità ciò che compone un pixel: una superficie è sottile, la nebbia è spessa |
| `floaters` | quota dell'opacità che sta dove non c'è niente: fuori dalla sagoma vera di una camera, o più di 4 cm davanti alla superficie vera |
| `needles` | quota delle Gaussiane visibili almeno tre volte più lunghe in una direzione che nella successiva |
| `opaque` | quota delle Gaussiane visibili con opacità sopra 0,5 |

Prima delle correzioni, nella demo su CPU, il 38% delle Gaussiane visibili
era ad ago e l'opacità mediana era 0,015. Ingrandita nel player, la scena
mostrava strisce e peli attorno al soggetto. Prova su un istante della demo
(240×135, CPU, 800 passi), camera esclusa e camere di training:

| Variante | Aghi | PSNR camera esclusa | PSNR training |
|---|---|---|---|
| senza correzioni | 38% | 21,6 dB | 39,6 dB |
| penalità sugli aghi 0,02 | 0% | 21,5 dB | 39,3 dB |
| dimensione minima 0,5 pixel | 13% | 22,1 dB | 37,7 dB |

La penalità toglie gli aghi quasi senza costo. La dimensione minima migliora
la camera esclusa e fa perdere nitidezza alle camere di training, che a
questa risoluzione erano adattate oltre il dovuto. Tutte e due sono attive di
default. Vanno misurate su GPU a piena risoluzione con la persona di prova.

Opzioni presenti ma spente, non ancora misurate a fondo: `lambda_opacity_entropy`
(una Gaussiana o c'è o non c'è), `lambda_ray_spread` (un pixel deve essere
fatto da una superficie e non da strati lontani in profondità), `init_flatten`
e `lambda_flat` (Gaussiane che partono e restano come dischi sulla superficie),
`opacity_decay_stop`. Le prime due, con pesi 0,05 e 0,1, dimezzano lo spessore
ma tolgono 7–10 dB alle camere di training: vanno provate con pesi dieci
volte più bassi.

## Idee recenti

Tra il 2025 e il 2026 sono usciti diversi lavori sul 4D Gaussian Splatting con
poche camere. Alcune loro idee sono state riscritte qui da zero, senza usare
il codice degli autori (vedi [LICENZE.md](LICENZE.md)), e provate una alla
volta. È rimasta quella che migliora la camera esclusa.

Prove sulla ripresa della demo (29 frame, 240×135, CPU, 1500 passi, `cam02`
esclusa), scena leggera (`--frame-step 3`). «Verifica» è la camera esclusa,
«training» la media delle tre usate. Il PSNR è misurato nel riquadro del
soggetto.

| Variante | Gaussiane | Verifica: PSNR / sfarfallio | Training: PSNR / sfarfallio | Esito |
|---|---|---|---|---|
| nessuna | 24 221 | 21,20 dB / 3,14 | 32,93 dB / 2,53 | |
| decadimento dell'opacità 0,005 | 22 222 | 21,44 dB / 3,04 | 32,77 dB / 2,56 | tenuto, default |
| decadimento dell'opacità 0,015 | 13 207 | 21,73 dB / 2,89 | 31,99 dB / 2,66 | disponibile |
| decadimento 0,015 solo finché dura la densificazione | 13 157 | 21,23 dB / 3,04 | 32,53 dB / 2,60 | scartato |
| vita a sommità piatta | 26 722 | 20,96 dB / 3,23 | 33,33 dB / 2,37 | scartato |
| vincolo tra frame consecutivi | 25 499 | 21,10 dB / 3,15 | 32,81 dB / 2,48 | scartato |
| opacità riveduta alla divisione | 24 142 | 21,15 dB / 3,13 | 32,87 dB / 2,53 | scartato |

Sulla scena di riferimento (una nuvola di partenza per frame) il decadimento
0,005 porta la verifica da 20,85 dB / 3,58 a 21,07 dB / 3,47. Il training
resta uguale (da 34,49 a 34,56 dB) e le Gaussiane passano da 65 368 a 64 679.

### Decadimento dell'opacità

Tenuto, attivo di default. L'idea viene da *4C4D: 4 Camera 4D Gaussian
Splatting* (Zhou e altri, CVPR 2026): con poche camere la geometria si impara
peggio dell'aspetto, e attenuare l'opacità delle Gaussiane aiuta la geometria.
In quell'articolo l'attenuazione è calcolata da una piccola rete. Anche
*FreeTimeGS* (Wang e altri, CVPR 2025) penalizza l'opacità. Qui dopo ogni
passo si toglie una quota fissa alle Gaussiane disegnate in quel passo.

Con 0,005 il guadagno è di 0,2 dB. È poco sopra il rumore di queste prove,
circa 0,1 dB, ma va nella stessa direzione su due scene, cresce con la quota
e lo sfarfallio cala. Con 0,015 le Gaussiane quasi si dimezzano e la camera
esclusa migliora ancora, ma le camere usate perdono 1 dB. Fermare il
decadimento quando finisce la densificazione annulla il guadagno.

In un training vero ogni Gaussiana viene disegnata circa cinque volte più
spesso che in queste prove, quindi la stessa quota avrà un effetto più forte.
Il valore va ricontrollato su GPU con riprese vere.

### Vita a sommità piatta

Scartata. *SharpTimeGS* (Liao e altri, CVPR 2026) sostituisce la campana che
regola l'opacità nel tempo con un profilo piatto dai fianchi ripidi. Qui è
stata provata alzando l'esponente della campana da 2 a 4. Le camere usate
migliorano di 0,4 dB e la camera esclusa peggiora di 0,24 dB. Avrebbe
richiesto anche di cambiare il formato, il player e il lettore per Unity.

### Vincolo tra frame consecutivi

Scartato. Si disegnano due frame consecutivi nello stesso passo e si chiede
che la loro differenza sia quella delle riprese. Nessun guadagno, e il
training è più lento del 70%.

### Opacità riveduta alla divisione

Scartata. È la correzione dell'opacità delle Gaussiane divise o duplicate
descritta in *Revising Densification in Gaussian Splatting* (Rota Bulò e
altri, ECCV 2024), già disponibile in gsplat. Nessun effetto.

### Colori delle camere

Tenuta come opzione. Compensare esposizione e bianco per camera è una pratica
nota. Prova su 5 frame a 96×72, con una camera resa più calda (rosso ×1,25,
blu ×0,80): con 450 passi i guadagni trovati per quella camera rispetto alla
vicina sono 1,21 e 0,86, e la differenza di tinta della scena tra i quattro
punti di vista scende da 0,15 a 0,06. Con 150 passi la correzione è a metà e
la tinta non cambia ancora. Serve quindi un training lungo. Non è stata
provata su camere vere.

### Open4DGS

Open4DGS (Arizona State University e Contrast Media Labs, novembre 2026) è una
pipeline completa per riprese con 27–45 camere. È fatta di due parti con
licenze diverse: la preparazione (Open4DGS-Core) è MIT, l'addestratore
(Open4DGS-Trigs) è AGPL-3.0. Dalla prima si può riusare il codice con la sua
nota di copyright, dalla seconda solo le idee (vedi [LICENZE.md](LICENZE.md)).
Finora non è stato copiato codice da nessuna delle due.

Preso da lì come idea:

- la scena intera, con una parte ferma e una in movimento, senza maschere nel
  training (qui: [Scena intera](#scena-intera-il-set-e-la-ripresa));
- i punti di partenza trovati confrontando le camere tra loro invece che dal
  solo visual hull. Open4DGS usa una rete di corrispondenza densa (RoMa v2),
  i cui pesi non hanno una licenza chiara per l'uso commerciale. Qui si usa
  la scansione a piani, che non richiede modelli.

Da prendere, in ordine:

1. la durata iniziale di una Gaussiana in base alla velocità: le parti lente
   nascono con una vita lunga e vengono seminate ogni pochi frame, le parti
   veloci a ogni frame. È il rimedio diretto alle Gaussiane che vivono un
   frame solo;
2. il riposizionamento delle Gaussiane poco opache al posto della
   densificazione, con un numero fisso di Gaussiane. gsplat ne ha già i
   mattoni, con licenza Apache-2.0;
3. il moto rigido continuo (rotazione e traslazione, con una curva di Bézier)
   dell'articolo TRiGS, per Gaussiane che seguono un arto per molti frame.
   Richiede di cambiare il formato del player e il lettore Unity;
4. la sincronizzazione sotto il frame e la stima del rolling shutter dalle
   immagini, per camere senza genlock.

La guida di Open4DGS segnala come problema aperto che i soggetti ricostruiti
restano un po' trasparenti. È lo stesso difetto misurato qui.

### Da provare

Il riposizionamento periodico delle Gaussiane poco opache (*FreeTimeGS*,
3DGS-MCMC). In queste prove il tetto di Gaussiane non viene raggiunto, quindi
non c'era nulla da ridistribuire. La densificazione che tiene conto del tempo
(*Temporally Aware Densification*, 2026) e la modulazione del moto con la
durata (*SharpTimeGS*). Per misurarle servono riprese con movimenti rapidi.

Riferimenti: Open4DGS, github.com/ASU-NEM-Labs/open4dgs-core e
open4dgs-trigs. TRiGS, arXiv 2604.00538. 4C4D, arXiv 2604.04063. FreeTimeGS, arXiv 2506.05348.
SharpTimeGS, arXiv 2602.02989. Temporally Aware Densification, arXiv
2606.23212.

## Formato del player

```
player.html                          tutto in un file (riprese brevi)
player/
  index.html, viewer.js, theme.css   player WebGL2 senza dipendenze
  scene.js, scene.json               descrizione: fps, frame, camere, blocchi
  chunks/0000.js, 0001.js, …         un blocco per spezzone della ripresa
```

Ogni Gaussiana è salvata una volta sola, con l'istante su cui è centrata, la
durata e la velocità. Il player la valuta nell'istante mostrato, come fa il
training: la sposta di
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
modalità per visori nel browser (WebXR): per il visore c'è il lettore Unity.

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

`tests/test_unity.py` copre quello che del lettore Unity si può verificare
senza Unity: gli stessi byte del player nella cartella per i motori grafici;
la parte in C# che legge i blocchi, sceglie e ordina le Gaussiane, compilata
ed eseguita con mono quando è installato (altrimenti il test viene saltato) e
confrontata con l'esportatore; i conti di proiezione dello shader, trascritti
e confrontati con la camera di riferimento.

`tests/test_launch.py` copre l'avvio di Studio: lettura e controllo delle
librerie richieste, riconoscimento di uno Studio già aperto, identità del
codice (cambia quando cambia un file del programma), pagina servita sempre
dalla versione in esecuzione e rifiuto di nuovi passi dopo un aggiornamento.

`tests/test_app.py` copre il server di MISTER 4D Studio: ogni comando che la
pagina può chiedere viene fatto leggere al parser della riga di comando (le
due interfacce non possono divergere senza che un test fallisca), poi coda dei
passi, interruzione, spiegazione degli errori e controlli di accesso (token,
indirizzo locale, file fuori dalla cartella del progetto).
