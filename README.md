# MISTER 4D Splatting

Proof of concept per ricostruire una persona in **4D** (Gaussian Splatting
dinamico) a partire dalle registrazioni di **3 o più camere fisse**, e
rivederla da nuovi punti di vista.

Segue l'impostazione della *Guida operativa set 4DGS* di MISTER Smart
Innovation: acquisizione sincronizzata, ricostruzione offline, riproduzione.
Questo repository copre la parte software dalla ricezione dei file video al
player.

```
video per camera → ingest → calibrate → prepare → masks → init → train → eval / render / export / lighten → view
```

| Passo | Che cosa fa |
|---|---|
| `ingest` | controlla i file, allinea le camere nel tempo (audio, flash o manuale), segnala i frame mancanti, estrae i frame |
| `calibrate` | intrinseci, distorsione ed estrinseci del rig da un pannello ChArUco; sistema mondo metrico; stima della copertura |
| `prepare` | undistort e ridimensionamento con intrinseci coerenti; fondale del set vuoto |
| `masks` | maschere soggetto/sfondo per confronto con il set vuoto |
| `init` | visual hull di ogni frame: geometria iniziale e ingombro del soggetto |
| `train` | ottimizzazione della scena dinamica su tutte le GPU disponibili; le riprese lunghe a spezzoni, con ripresa dopo un'interruzione; su richiesta impara gli scarti di sync tra camere |
| `eval` | metriche (qualità, sagoma, sfarfallio) e immagini di confronto su una camera esclusa dal training |
| `render` | video del risultato: anteprima, oppure materiale per il montaggio (ripresa intera, qualunque risoluzione, sfondo trasparente in ProRes 4444 o PNG, movimento di camera a piacere) |
| `export` | player 4D in **un solo file HTML** che si apre con doppio clic; per le riprese lunghe una cartella che carica la scena a pezzi; la scena per Unity; oppure sequenza di `.ply` |
| `lighten` | copia più leggera di una ripresa già ricostruita, per i visori autonomi |

Tutto gira **in locale** sulla propria macchina: non c'è nessun servizio web o
cloud, e nessun dato lascia il computer. Il player è una pagina HTML letta dal
browser direttamente da disco.

Chi non usa il terminale lavora con **[MISTER 4D Studio](#mister-4d-studio-senza-terminale)**:
una pagina guidata che esegue gli stessi passi, nello stesso ordine, e ne
spiega l'esito in italiano. Su Windows si apre con doppio clic.

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
- ricostruzione dinamica: vedi [Risultato della demo](#risultato-della-demo);
- punti di vista mai visti dalle camere: il decadimento dell'opacità, un'idea
  presa da lavori del 2025–2026 e riscritta qui, migliora la camera esclusa
  di 0,2 dB su due scene di prova e ne riduce lo sfarfallio, con qualche
  Gaussiana in meno. Altre tre idee sono state misurate e scartate (tabella
  in [docs/PIPELINE.md](docs/PIPELINE.md), "Idee recenti");
- scarti di sincronizzazione sotto il frame: con due camere sfasate di +0,40 e
  −0,30 frame il training ne ha stimati +0,18 e −0,31 e la qualità è salita
  (IoU della sagoma dalla camera esclusa da 0,78 a 0,80): segno e ordine di
  grandezza giusti, valore non sempre esatto.

**Non ancora verificato** (lo sviluppo è avvenuto su una macchina Linux senza
GPU e senza riprese reali):

- lo script di installazione per Windows: le versioni che installa esistono
  sugli indici ufficiali di PyTorch e gsplat, e i test passano su Linux con
  le stesse versioni (Python 3.10, PyTorch 2.4.1, NumPy 1.26, gsplat 1.5.3),
  ma lo script non è stato eseguito su una macchina Windows;
- MISTER 4D Studio su Windows: è stato collaudato in un browser su Linux
  dall'inizio alla fine (riprese di esempio, i sei passi, interruzione di un
  passo, chiusura) e il suo server ha test automatici, ma il file di avvio
  `MISTER 4D Studio.bat` e la finestra «Sfoglia…» non sono stati provati su
  una macchina Windows;
- le riprese lunghe, le scene leggere e la versione per visore: i
  meccanismi hanno test automatici e misure su dati sintetici minuscoli (vedi
  [Riprese lunghe, scene leggere, montaggio, visori](#riprese-lunghe-scene-leggere-montaggio-visori)),
  ma non sono mai stati usati su una ripresa vera;
- il decadimento dell'opacità e la compensazione dei colori tra camere:
  misurati solo su prove minuscole su CPU. L'intensità del primo va
  ricontrollata su GPU (in un training vero agisce più a lungo); la seconda
  non ha mai visto camere vere;
- il lettore per Unity: la lettura dei dati e l'ordinamento sono stati
  compilati ed eseguiti fuori da Unity, e i conti dello shader confrontati con
  quelli del player, ma né il componente né lo shader sono mai stati aperti in
  Unity, e nessuna scena è stata vista su un visore;
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

## MISTER 4D Studio (senza terminale)

Per chi deve usare il programma senza scrivere comandi. Su Windows, dopo
l'installazione: doppio clic su **`MISTER 4D Studio.bat`** nella cartella del
progetto. Si apre una pagina nel browser; la finestra nera che resta aperta è
il programma e si chiude premendo **Esci** nella pagina. Da terminale, su
qualunque sistema:

```bash
mister4d app
```

Una spiegazione non tecnica di tutto il percorso, da dare a chi lo usa:
[docs/COME_FUNZIONA.md](docs/COME_FUNZIONA.md).

La pagina accompagna attraverso sei passi, nell'ordine in cui vanno fatti. In
ogni passo si legge prima che cosa serve, poi si preme un solo pulsante, poi
si legge che cosa è venuto fuori; le opzioni che servono di rado sono
richiuse in «Opzioni avanzate».

| | Passo | Che cosa chiede | Che cosa restituisce |
|---|---|---|---|
| 01 | Progetto | la cartella dove salvare il lavoro | stato del computer: schede grafiche, FFmpeg, spazio libero |
| 02 | Video | le tre cartelle delle riprese e come sono sincronizzate | durata, fotogrammi, allineamento e fotogrammi persi per camera |
| 03 | Calibrazione | le misure del pannello, in centimetri | errore in pixel con un giudizio, copertura delle camere |
| 04 | Preparazione | la risoluzione di lavoro | anteprima delle maschere, da controllare prima di proseguire |
| 05 | Ricostruzione | la camera di verifica (facoltativa), il peso della scena | avanzamento con tempo residuo, poi riepilogo della scena |
| 06 | Risultato | quali materiali in più creare (facoltativo) | player, video dimostrativo, materiali richiesti e, con la camera di verifica, qualità misurata |

- Le riprese lunghe vengono ricostruite a spezzoni, senza nulla da
  impostare: l'avanzamento li conta, e se il lavoro viene interrotto
  ripetendo il passo riparte dal primo spezzone mancante.
- Nel passo Ricostruzione si sceglie il **peso della scena** (massimo
  dettaglio, bilanciata, leggera); nel passo Risultato, sotto «Materiale per
  il montaggio e per il visore», si possono chiedere il video del soggetto su
  sfondo trasparente, la versione leggera per visore autonomo e la sequenza
  di file PLY.
- Sempre nel passo Ricostruzione, due caselle per camere non perfette:
  una fa stimare gli scarti di tempo tra camere senza sincronizzazione
  hardware, l'altra le differenze di colore tra camere che non rendono i
  colori allo stesso modo.
- Accanto a ogni richiesta una piccola «i» apre la sua spiegazione: che
  cos'è, come prepararla, che cosa succede se è sbagliata. La voce «Come
  funziona questo passo» racconta il resto; nel passo Calibrazione contiene
  anche il pulsante che crea il file del pannello da stampare.
- Studio **non è una seconda implementazione**: ogni passo lancia i comandi
  `mister4d` descritti più sotto, compilandone gli argomenti e traducendone
  l'esito. Quello che si ottiene dalla pagina e dal terminale è identico.
- Lo stato sta nella cartella del progetto: chiudendo e riaprendo si riparte
  da dove si era. Cambiare un'opzione riapre i passi che ne dipendono, e solo
  quelli.
- Gli errori più comuni sono spiegati in italiano («il pannello è stato
  riconosciuto in troppo pochi fotogrammi…»); il testo tecnico completo resta in
  «Dettagli tecnici» e nella cartella `registro/`.
- «Crea le riprese di esempio», nel passo Video, permette di provare tutto il
  percorso senza camere.
- Su un computer senza scheda grafica NVIDIA la ricostruzione viene fatta in
  versione ridotta (le stesse impostazioni di `mister4d demo --run` senza
  GPU) e la pagina lo dice.
- Nulla esce dal computer: la pagina risponde solo all'indirizzo locale
  `127.0.0.1` e solo a se stessa.
- Pagina e programma sono sempre della stessa versione. Il file di avvio fa
  caricare a Python il codice della cartella in cui si trova, qualunque copia
  sia installata; a ogni avvio il programma installa le librerie che un
  aggiornamento può aver aggiunto (senza toccare PyTorch e gsplat) e, se è già
  aperto, mostra quello aperto invece di avviarne un secondo. Se i file
  cambiano mentre è aperto (un `git pull`), la pagina lo dice e non parte
  nessun nuovo passo finché non viene riaperto; una pagina rimasta aperta
  dopo la riapertura si ricarica da sola.

Cartella di un progetto:

```
progetto.json     scelte fatte e stato dei passi
lavoro/           fotogrammi estratti e calibrazione
dati/             immagini corrette, maschere, punti iniziali
risultato/        player.html, orbita.mp4 e run/ (scena e misure)
registro/         testo completo di ogni passo
```

I colori di Studio e del player sono quelli che MISTER Smart Innovation usa
in pubblico: il blu del logo, e il navy, il grigio ghiaccio, l'inchiostro e
il rosso mattone del sito labmister.it. Stanno in un solo file,
`src/mister4d/viewer/theme.css`: cambiando i cinque valori `--mister-*`
cambiano entrambe le applicazioni. Il logo non è incluso nel repository.

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

| | PSNR soggetto | IoU sagoma | Sfarfallio |
|---|---|---|---|
| camera esclusa (nuovo punto di vista) | 21,5 dB | 0,87 | 3,3 |
| camere di training | 35,3 dB | 0,99 | 1,8 |

Prima del decadimento dell'opacità la camera esclusa era a 21,2 dB con
sfarfallio 3,4; le camere di training non sono cambiate.

Sono numeri di una prova minuscola su CPU: servono a dire che la catena
funziona da un capo all'altro, non a prevedere la qualità su riprese reali.
Lo scarto tra le due righe è quasi tutto geometria: tolta `cam02`, le tre
camere rimaste lasciano un angolo scoperto di circa 100° proprio dove guarda
lei, e da lì il soggetto può apparire fino al 60% più largo (lo dice il
programma stesso: vedi "Copertura delle camere" in
[docs/PIPELINE.md](docs/PIPELINE.md)). Con tutte e quattro le camere lo stesso
calcolo dà il 10%.

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
- `export --format web` scrive invece una cartella che carica la scena a
  pezzi, per le riprese lunghe: si apre con doppio clic su `index.html`.
  `--format ply` scrive un `.ply` 3DGS per frame.
- Ogni comando ha `--help`; tutte le opzioni di training sono in
  `configs/default.yaml` e si cambiano con `--set chiave=valore`.

## Riprese lunghe, scene leggere, montaggio, visori

**Riprese lunghe.** Non c'è nulla da impostare: una ripresa più lunga di circa
tre secondi viene ricostruita a spezzoni di due secondi, ognuno un training a
sé (`runs/take01/seg_000`, `seg_001`, …). Tempo e memoria di ogni training non
dipendono dalla durata; il totale cresce in proporzione. Se il lavoro si
interrompe, lo stesso comando lo riprende dal primo spezzone mancante
(`--fresh` per rifare tutto). Il player di una ripresa lunga è una cartella:

```bash
mister4d export runs/take01 --format web --out exports/take01
```

**Scene leggere.** Con una nuvola di partenza ogni tre frame invece che a
ogni frame le Gaussiane vivono tre frame e si muovono con il soggetto; ne
servono circa un terzo:

```bash
mister4d init data/take01 --frame-step 3
```

```bash
mister4d train data/take01 --out runs/take01_leggera --set max_gaussians=660000
```

**Versione per visore autonomo.** Da una ripresa già ricostruita, una copia
con al massimo tante Gaussiane visibili insieme quante ne regge il visore:

```bash
mister4d lighten runs/take01 --out runs/take01_visore --budget 80000
```

```bash
mister4d export runs/take01_visore --format engine --out exports/scena_visore
```

La cartella `scena_visore` è quella che legge il componente per **Unity** in
[unity/](unity/README.md), pensato per Quest 3: si copia in
`Assets/StreamingAssets` del progetto. Con `--format web` si ottiene invece la
stessa scena da guardare nel browser. Il componente Unity **non è ancora stato
aperto in Unity né provato su un visore**: in [unity/README.md](unity/README.md)
c'è che cosa è stato verificato lo stesso e che cosa controllare alla prima
apertura. Per Unreal non c'è un lettore; il formato è descritto nello stesso
documento.

**Materiale per il montaggio.** La ripresa intera con il solo soggetto su
sfondo trasparente, in ProRes 4444 oppure come immagini PNG, alla risoluzione
che serve:

```bash
mister4d render runs/take01 --out exports/soggetto.mov --take --transparent --height 2160
```

```bash
mister4d render runs/take01 --out exports/soggetto_png --take --transparent
```

Per un movimento di camera proprio: nel player si sceglie un punto di vista,
«Copia la vista», e si incollano le viste in un file JSON (una lista, ognuna
con il suo `time` in secondi); poi `mister4d render … --path viste.json`.

Misure sulla ripresa sintetica della demo (29 frame, 240×135, CPU, 1500 passi
per training, `cam02` esclusa e usata come verifica). «Verifica» è la camera
mai vista, «training» la media delle tre usate; PSNR nel riquadro del
soggetto.

| Variante | Gaussiane | Player | Verifica: PSNR / IoU / sfarfallio | Training: PSNR / sfarfallio |
|---|---|---|---|---|
| di riferimento, un pezzo | 65 368 | 1,0 MB | 20,9 dB / 0,86 / 3,6 | 34,5 dB / 1,9 |
| di riferimento, in 3 spezzoni | 81 603 | 1,2 MB | 20,7 dB / 0,86 / 3,8 (4,3 dove due spezzoni si incontrano) | 35,9 dB / 1,6 (1,6 dove si incontrano) |
| bilanciata (`--frame-step 2`) | 34 485 | 0,6 MB | 21,3 dB / 0,87 / 3,2 | 33,3 dB / 2,4 |
| leggera (`--frame-step 3`) | 24 228 | 0,4 MB | 21,2 dB / 0,87 / 3,1 | 32,9 dB / 2,5 |
| per visore (`lighten`, metà delle Gaussiane visibili insieme) | 6 108 | 0,2 MB | 20,7 dB / 0,86 / 3,4 | 32,6 dB / 2,5 |

Queste misure sono state fatte prima di introdurre il decadimento
dell'opacità, che ora è attivo di default: con esso la scena di riferimento
passa a 64 679 Gaussiane e 21,1 dB / 0,86 / 3,5 in verifica (34,6 dB / 1,9 in
training), la leggera a 22 222 Gaussiane e 21,4 dB / 0,87 / 3,0 (32,8 dB /
2,6). Le altre righe non sono state rimisurate.

Nel formato precedente, che riscriveva le Gaussiane a ogni frame, il player
della scena di riferimento pesava 3,4 MB. Sono numeri di una prova minuscola,
con un soggetto che si muove lentamente: dicono che i meccanismi funzionano e
in che direzione vanno i compromessi, non quanto pesa o come rende una
ripresa vera. In particolare la scena leggera può perdere nitidezza con
movimenti rapidi, e nessuna di queste scene è stata provata su un visore.

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
  vengono inventate. `render` percorre solo l'arco coperto dal rig. Tra una
  camera e l'altra il soggetto appare più largo del vero, tanto più quanto le
  camere sono distanti: `calibrate` e `train` dicono di quanto (dettagli in
  [docs/PIPELINE.md](docs/PIPELINE.md), "Copertura delle camere").
- **Sincronizzazione.** `ingest` allinea al frame più vicino e riporta l'errore
  residuo. Lo scarto sotto il frame si può far stimare al training
  (`--set learn_time_offsets=true`), ma è una stima: per gesti rapidi resta
  preferibile la sincronizzazione hardware.
- **Colori.** Camere con esposizione o bianco diversi danno una scena che
  cambia tinta con il punto di vista. Il training può stimare un guadagno per
  camera (`--set learn_camera_colors=true`), ma è una stima che richiede un
  training lungo e non è stata provata su camere vere: resta preferibile
  bloccare esposizione e bianco su tutte.
- **Maschere.** Dipendono dal set vuoto: luce o inquadratura che cambiano tra
  le due riprese le rovinano; le ombre nette a terra possono entrare nella
  maschera. Si possono fornire maschere fatte con altri strumenti.
- **Coerenza temporale.** Le Gaussiane non sono le stesse da un istante
  all'altro: da punti di vista lontani dalle camere possono restare piccoli
  sfarfallii, e non c'è una traiettoria per punto utilizzabile per analisi del
  gesto.
- **Durata.** Le riprese lunghe vengono ricostruite a spezzoni di due secondi:
  tempo di calcolo, spazio su disco e peso del player crescono in proporzione
  alla durata (250 passi di training per frame). Il meccanismo è collaudato
  solo su dati sintetici minuscoli: tempi e peso su riprese vere, e la resa
  nei punti in cui due spezzoni si incontrano, vanno misurati sulla
  workstation.
- **Player.** Colore indipendente dalla vista, sia nel browser sia in Unity.
- **Visore.** Il lettore per Unity (Quest 3) è scritto e verificato solo nelle
  parti che non richiedono Unity: **non è mai stato aperto in Unity né provato
  su un visore**. Per Unreal non c'è un lettore.
- **Nessuna misura.** Fotorealismo non significa accuratezza metrica: il
  risultato non è una superficie su cui prendere quote.

## Struttura

```
MISTER 4D Studio.bat   avvio di Studio con doppio clic (Windows)
unity/            componente per Unity (Quest 3) e descrizione del formato per altri motori
src/mister4d/     pacchetto Python (un modulo per passo della pipeline)
  run.py          lettura di una ripresa ricostruita, intera o a spezzoni
  lighten.py      copia leggera per visori autonomi
  launch.py       avvio di Studio: librerie, istanza unica, versione
  app.py          MISTER 4D Studio: server locale che guida i passi
  studio/         pagina di MISTER 4D Studio
  viewer/         player WebGL2 e colori condivisi (theme.css)
configs/          opzioni di training
scripts/          installazione (Windows e Linux), avvio di Studio (Linux)
tests/            test automatici
docs/             come funziona (non tecnico), licenze, pipeline, sessione di ripresa
```

Test:

```bash
pytest
```

## Obiettivo: il livello dei lavori premiati a Venezia

Il riferimento sono i lavori con attori in 4D Gaussian Splatting presentati a
Venice Immersive 2026, come *Empire At Sea* (Achievement Prize): 22 minuti,
attori ripresi su un palco con 100 camere 8K, ricostruzione affidata a
un'azienda specializzata, riproduzione su visore. Rispetto a quel livello
mancano, in ordine di peso:

1. **Camere.** Con 3–4 camere il programma deve indovinare tutto ciò che sta
   tra un punto di vista e l'altro, e nessun software lo compensa. Un
   obiettivo intermedio realistico sono 16–24 camere sincronizzate via
   hardware, con luce uniforme.
2. **Durata.** La ricostruzione a spezzoni e il player che carica la scena a
   pezzi ci sono, ma sono collaudati solo su dati sintetici minuscoli: vanno
   misurati tempi, peso e resa su riprese vere di decine di secondi.
3. **Resa.** Colore che cambia con il punto di vista, capelli e mani, bordi
   puliti (maschere migliori), stabilità da un fotogramma all'altro. Le idee
   dei lavori più recenti vengono provate una alla volta e tenute solo se
   misurate utili ([docs/PIPELINE.md](docs/PIPELINE.md), "Idee recenti"): le
   prossime, da provare su GPU con riprese vere, sono il riposizionamento
   delle Gaussiane e la densificazione che tiene conto del tempo.
4. **Riproduzione.** Un visore o un motore grafico al posto della pagina web.
   La versione leggera della scena e un lettore per Unity pensato per Quest 3
   ([unity/](unity/README.md)) ci sono; il lettore va aperto in Unity e provato
   sul visore, e solo lì si vedrà quante Gaussiane regge e come rende.

Prima di tutto questo viene il collaudo su GPU e su riprese reali, che non è
ancora stato fatto.

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
