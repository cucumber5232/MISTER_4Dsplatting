# MISTER 4D Splatting

Ricostruisce una persona in 4D (Gaussian Splatting dinamico) dalle riprese di
tre o più camere fisse, per rivederla da altri punti di vista.

Segue la *Guida operativa set 4DGS* di MISTER Smart Innovation e copre la
parte software, dai file video al player.

```
video per camera → ingest → calibrate → prepare → masks → init → train → eval / render / export / lighten → view
```

| Passo | Che cosa fa |
|---|---|
| `ingest` | controlla i file, allinea le camere nel tempo (audio, flash o a mano), segnala i frame mancanti, estrae i frame |
| `calibrate` | intrinseci, distorsione ed estrinseci da un pannello ChArUco, in metri |
| `prepare` | toglie la distorsione e ridimensiona; prepara il fondale dal set vuoto |
| `masks` | separa il soggetto dallo sfondo confrontando con il set vuoto |
| `init` | visual hull di ogni frame, come geometria di partenza |
| `train` | ottimizza la scena su tutte le GPU; le riprese lunghe a spezzoni |
| `eval` | misura qualità, sagoma e sfarfallio su una camera tenuta fuori |
| `render` | video del risultato, anche con sfondo trasparente per il montaggio |
| `export` | player in un file HTML, cartella per le riprese lunghe, scena per Unity, sequenza `.ply` |
| `lighten` | copia più leggera della scena per i visori autonomi |

Gira tutto sul computer locale, senza servizi esterni. Il player è una pagina
HTML che si apre da disco.

Chi non usa il terminale lavora con [MISTER 4D Studio](#mister-4d-studio), una
pagina guidata che esegue gli stessi passi. Su Windows si apre con doppio
clic.

## Licenza

Il codice di questo repository è di proprietà di MISTER Smart Innovation,
tutti i diritti riservati. Il testo è nel file [LICENSE](LICENSE).

È scritto da zero e usa solo librerie con licenze permissive: gsplat
(Apache-2.0) per la rasterizzazione su GPU, PyTorch, OpenCV, NumPy, SciPy,
PyYAML. Non usa il codice Inria di 3D Gaussian Splatting, che ha una licenza
di sola ricerca, né i progetti che ne dipendono, né modelli corporei come
SMPL. L'elenco completo è in [docs/LICENZE.md](docs/LICENZE.md).

## Stato

Lo sviluppo è avvenuto su una macchina Linux senza GPU e senza riprese vere.
Le verifiche sono state fatte su dati sintetici, con i test automatici e con
la demo.

Verificato:

- sincronizzazione: gli scarti di avvio tra le camere vengono ritrovati
  esattamente, da audio e da flash. Un frame perso viene segnalato;
- calibrazione: errore di riproiezione 0,12 px, camere entro 5 mm e focali
  entro lo 0,2% dai valori veri;
- ricostruzione: i numeri sono in [Risultato della demo](#risultato-della-demo);
- scarti di tempo sotto il frame: con due camere sfasate di +0,40 e −0,30
  frame il training stima +0,18 e −0,31.

Da verificare sulla workstation:

- installazione e avvio su Windows (`install.bat`, `MISTER 4D Studio.bat`, la
  finestra «Sfoglia…»). Studio è stato provato in un browser su Linux;
- gsplat su GPU e training su due GPU. Il primo comando da lanciare è
  `mister4d doctor`;
- riprese lunghe, scene leggere e versione per visore su una ripresa vera;
- decadimento dell'opacità e compensazione dei colori su riprese vere;
- la ricostruzione della scena intera (`--full-scene`): i test automatici
  passano e una prova a 240×135 su CPU funziona da capo a fondo, ma non è mai
  stata addestrata su GPU a piena risoluzione;
- le correzioni contro le Gaussiane ad ago e contro la nebbia (vedi
  [docs/PIPELINE.md](docs/PIPELINE.md), «Qualità»): provate solo su CPU a
  bassa risoluzione, vanno misurate su GPU con la persona di prova;
- il lettore per Unity, mai aperto in Unity né provato su un visore;
- qualità e tempi su riprese reali. I valori di default sono un punto di
  partenza.

## Installazione

Servono FFmpeg e, per il training, una scheda NVIDIA con i driver aggiornati.

### Windows

Servono Python 3.10 (da python.org, oppure `winget install Python.Python.3.10`),
Git e FFmpeg (`winget install Gyan.FFmpeg`). Visual Studio e CUDA Toolkit non
servono.

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

Lo script crea l'ambiente `.venv`, installa PyTorch 2.4.1 per CUDA 12.4,
gsplat 1.5.3 già compilato e il progetto, poi lancia `mister4d doctor`. Python
deve essere il 3.10 perché gsplat già compilato per Windows esiste solo per
quella versione. Con altre versioni gsplat va compilato a mano (istruzioni in
`docs/INSTALL_WIN.md` nel repository di gsplat).

La cartella del progetto deve avere un percorso senza lettere accentate,
perché su Windows OpenCV non apre i file che ne contengono.

In ogni nuovo terminale l'ambiente va attivato dalla cartella del progetto:

```bat
.venv\Scripts\activate.bat
```

### Linux

Serve Python 3.10 o successivo.

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

Con l'ambiente attivo il programma è il comando `mister4d`. Questo controlla
l'installazione e prova le GPU:

```bash
mister4d doctor
```

## MISTER 4D Studio

Per chi usa il programma senza terminale. Su Windows, dopo l'installazione,
doppio clic su `MISTER 4D Studio.bat` nella cartella del progetto. Si apre una
pagina nel browser. La finestra nera che resta aperta è il programma e si
chiude con «Esci» nella pagina. Da terminale:

```bash
mister4d app
```

[docs/COME_FUNZIONA.md](docs/COME_FUNZIONA.md) spiega tutto il percorso senza
termini tecnici, per chi lo usa.

La pagina propone sei passi, nell'ordine in cui vanno fatti.

| | Passo | Che cosa chiede | Che cosa restituisce |
|---|---|---|---|
| 01 | Progetto | la cartella dove salvare il lavoro | stato del computer: schede grafiche, FFmpeg, spazio libero |
| 02 | Video | le tre cartelle delle riprese e come sono sincronizzate | durata, fotogrammi, allineamento e fotogrammi persi per camera |
| 03 | Calibrazione | le misure del pannello, in centimetri | errore in pixel con un giudizio, copertura delle camere |
| 04 | Preparazione | la risoluzione di lavoro | anteprima delle maschere, da controllare prima di proseguire |
| 05 | Ricostruzione | la camera di verifica (facoltativa), il peso della scena | avanzamento con tempo residuo, poi riepilogo della scena |
| 06 | Risultato | quali materiali in più creare (facoltativo) | player, video dimostrativo, materiali richiesti e qualità misurata se c'è la camera di verifica |

Altre cose da sapere:

- Una ripresa lunga viene ricostruita a spezzoni. Se il lavoro si interrompe,
  ripetendo il passo riparte dal primo spezzone mancante.
- Nel passo Ricostruzione si sceglie il peso della scena (massimo dettaglio,
  bilanciata, leggera). Ci sono anche due caselle: una per camere senza
  sincronizzazione hardware, una per camere che rendono i colori in modo
  diverso.
- Nel passo Risultato si possono chiedere il video del soggetto su sfondo
  trasparente, la versione leggera per visore e i file PLY.
- La «i» accanto a ogni richiesta apre la sua spiegazione. «Come funziona
  questo passo» spiega il resto. Nel passo Calibrazione c'è il pulsante che
  crea il pannello da stampare.
- Ogni passo lancia i comandi `mister4d` descritti più sotto, quindi il
  risultato è lo stesso che da terminale.
- Lo stato è salvato nella cartella del progetto. Chiudendo e riaprendo si
  riparte da dove si era. Se si cambia un'opzione vanno rifatti solo i passi
  che ne dipendono.
- Gli errori più comuni sono spiegati in italiano. Il testo tecnico completo
  è in «Dettagli tecnici» e nella cartella `registro/`.
- «Crea le riprese di esempio», nel passo Video, permette di provare tutto
  senza camere.
- Senza scheda NVIDIA la ricostruzione viene fatta in versione ridotta, e la
  pagina lo dice.
- La pagina risponde solo all'indirizzo locale `127.0.0.1`.
- A ogni avvio il programma installa le librerie che un aggiornamento può
  aver aggiunto, senza toccare PyTorch e gsplat. Se è già aperto mostra la
  finestra aperta. Se i file cambiano mentre è aperto, per esempio dopo un
  `git pull`, la pagina chiede di chiuderlo e riaprirlo.

Cartella di un progetto:

```
progetto.json     scelte fatte e stato dei passi
lavoro/           fotogrammi estratti e calibrazione
dati/             immagini corrette, maschere, punti iniziali
risultato/        player.html, orbita.mp4 e run/ (scena e misure)
registro/         testo completo di ogni passo
```

I colori di Studio e del player sono quelli di MISTER Smart Innovation: il blu
del logo, il navy, il grigio ghiaccio, l'inchiostro e il rosso mattone del
sito labmister.it. Sono in `src/mister4d/viewer/theme.css`, nei cinque valori
`--mister-*`. Il logo non è nel repository.

## Prova senza camere

Questo comando crea una sessione di ripresa finta (manichino animato, quattro
camere con distorsione, avvii sfalsati, audio con battito di mani, pannello
ChArUco, set vuoto) ed esegue tutta la pipeline, dai video al player:

```bash
mister4d demo --out sessions/demo --run
```

Con la GPU lavora a piena risoluzione. Senza GPU usa immagini 240×135 e il
rasterizzatore su CPU, e dura circa un quarto d'ora. Alla fine si apre
`sessions/demo/player.html` con doppio clic.

Il manichino serve a provare la pipeline in fretta. Per giudicare quanto il
risultato sembra vero c'è una persona vestita, con mani, viso, capelli, stampa
sulla felpa, su un set con pavimento a doghe, lampada e ombra a terra:

```bash
mister4d demo --out sessions/demo_persona --subject human --run
```

La figura deriva dai dati CC0 di MakeHuman (vedi [LICENZE.md](docs/LICENZE.md)).
In due secondi sposta il peso, alza un braccio e saluta, gira la testa e fa un
passo. Il gesto del braccio è veloce: la mano supera i 2 m/s.

### Risultato della demo

Su CPU, 240×135 px, 29 frame, training su tre camere (`cam01`, `cam03`,
`cam04`), `cam02` tenuta fuori per la verifica.

| | PSNR soggetto | IoU sagoma | Sfarfallio |
|---|---|---|---|
| camera esclusa | 21,5 dB | 0,87 | 3,3 |
| camere di training | 35,3 dB | 0,99 | 1,8 |

È una prova piccola e serve a controllare che tutta la catena funzioni. La
differenza tra le due righe dipende soprattutto dalla posizione delle camere:
senza `cam02` resta un angolo scoperto di circa 100°, e da lì il soggetto può
apparire fino al 60% più largo. Con tutte e quattro le camere il valore scende
al 10%. Il calcolo è spiegato in [docs/PIPELINE.md](docs/PIPELINE.md), alla
voce «Copertura delle camere».

## Uso con riprese vere

Che cosa registrare e come è in
[docs/SESSIONE_DI_RIPRESA.md](docs/SESSIONE_DI_RIPRESA.md). Servono tre clip,
ognuna con un video per camera e gli stessi nomi di file (`cam01.mp4`,
`cam02.mp4`, …): pannello ChArUco, set vuoto, soggetto.

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

Prima di proseguire va guardato `data/take01/masks_preview.jpg`.

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

`exports/take01.html` contiene player e dati. Si apre con doppio clic e si può
consegnare così.

Note:

- I comandi sono gli stessi su Windows e Linux. A `ingest` si passa la
  cartella della clip, e prende tutti i video che contiene.
- `--sync` vale `none` per camere con trigger o genlock, `audio` per il
  battito di mani, `flash`, oppure si danno gli scarti a mano con
  `--offsets cam02=3,…`. `--start N` scarta i primi N frame.
- `--square` e `--marker` sono le dimensioni misurate sul pannello stampato.
- La camera indicata con `--holdout` e `--exclude` resta fuori dal training e
  serve a misurare la qualità. Per il risultato finale si rilancia senza.
- `export --format web` scrive una cartella per le riprese lunghe, che si apre
  con doppio clic su `index.html`. `--format ply` scrive un `.ply` per frame.
- Ogni comando ha `--help`. Le opzioni di training sono in
  `configs/default.yaml` e si cambiano con `--set chiave=valore`.

## Riprese lunghe, scene leggere, montaggio, visore

### Riprese lunghe

Una ripresa più lunga di circa tre secondi viene ricostruita a spezzoni di due
secondi, ognuno con il suo training (`runs/take01/seg_000`, `seg_001`, …).
Tempo e memoria di ogni spezzone non dipendono dalla durata della ripresa. Se
il lavoro si interrompe, lo stesso comando riparte dal primo spezzone mancante
(`--fresh` per rifare tutto). Il player di una ripresa lunga è una cartella:

```bash
mister4d export runs/take01 --format web --out exports/take01
```

### Scena intera

Di norma il soggetto viene ritagliato dal fondo con le maschere. Con
`--full-scene` si ricostruisce tutto, set compreso, e le maschere non
servono. Prima viene ricostruito il set vuoto, come scena ferma. Poi sopra si
ricostruisce quello che si muove, confrontando le immagini intere.

```bash
mister4d init data/take01 --full-scene --exclude cam02
```

```bash
mister4d train data/take01 --out runs/take01_intera --full-scene --holdout cam02
```

Il passo `masks` si salta. Se il set vuoto non è stato registrato viene
stimato dalla ripresa, e quello che resta fermo per tutta la ripresa finisce
nel set. Con poche camere il set viene bene solo dai punti di vista vicini a
quelli ripresi. Per provarlo senza camere:

```bash
mister4d demo --out sessions/demo_intera --subject human --full-scene --run
```

### Scene leggere

Con una nuvola di partenza ogni tre frame le Gaussiane vivono tre frame e si
muovono con il soggetto. Ne serve circa un terzo.

```bash
mister4d init data/take01 --frame-step 3
```

```bash
mister4d train data/take01 --out runs/take01_leggera --set max_gaussians=660000
```

### Visore autonomo

Da una ripresa già ricostruita si ottiene una copia con un numero massimo di
Gaussiane visibili insieme, scelto in base a quante ne regge il visore:

```bash
mister4d lighten runs/take01 --out runs/take01_visore --budget 80000
```

```bash
mister4d export runs/take01_visore --format engine --out exports/scena_visore
```

La cartella `scena_visore` va copiata in `Assets/StreamingAssets` di un
progetto Unity e la legge il componente in [unity/](unity/README.md), scritto
per Quest 3. Il componente non è ancora stato aperto in Unity né provato su un
visore: in [unity/README.md](unity/README.md) c'è l'elenco delle cose da
controllare alla prima apertura. Per Unreal non c'è un lettore, ma lo stesso
documento descrive il formato.

### Materiale per il montaggio

La ripresa con il solo soggetto su sfondo trasparente, in ProRes 4444 o come
immagini PNG, alla risoluzione che serve:

```bash
mister4d render runs/take01 --out exports/soggetto.mov --take --transparent --height 2160
```

```bash
mister4d render runs/take01 --out exports/soggetto_png --take --transparent
```

Per un movimento di camera scelto a mano si sceglie un punto di vista nel
player, si preme «Copia la vista» e si incollano le viste in un file JSON (una
lista, ognuna con il suo `time` in secondi). Poi
`mister4d render … --path viste.json`.

### Misure

Sulla ripresa della demo (29 frame, 240×135, CPU, 1500 passi, `cam02`
esclusa). «Verifica» è la camera esclusa, «training» la media delle tre usate.
Il PSNR è misurato nel riquadro del soggetto.

| Variante | Gaussiane | Player | Verifica: PSNR / IoU / sfarfallio | Training: PSNR / sfarfallio |
|---|---|---|---|---|
| di riferimento, un pezzo | 65 368 | 1,0 MB | 20,9 dB / 0,86 / 3,6 | 34,5 dB / 1,9 |
| di riferimento, in 3 spezzoni | 81 603 | 1,2 MB | 20,7 dB / 0,86 / 3,8 | 35,9 dB / 1,6 |
| bilanciata (`--frame-step 2`) | 34 485 | 0,6 MB | 21,3 dB / 0,87 / 3,2 | 33,3 dB / 2,4 |
| leggera (`--frame-step 3`) | 24 228 | 0,4 MB | 21,2 dB / 0,87 / 3,1 | 32,9 dB / 2,5 |
| per visore (`lighten`, metà delle Gaussiane visibili) | 6 108 | 0,2 MB | 20,7 dB / 0,86 / 3,4 | 32,6 dB / 2,5 |

Queste righe sono state misurate prima di aggiungere il decadimento
dell'opacità, ora attivo di default. Con il decadimento la scena di
riferimento ha 64 679 Gaussiane e 21,1 dB / 0,86 / 3,5 in verifica, la leggera
22 222 Gaussiane e 21,4 dB / 0,87 / 3,0. Le altre righe non sono state
rimisurate.

La prova è piccola e il soggetto si muove lentamente. Con movimenti rapidi la
scena leggera può perdere nitidezza. Nessuna di queste scene è stata provata
su un visore.

## Scena di prova sulla workstation

`Scena di prova su GPU.bat`, nella cartella del progetto, fa tutto da solo sul
computer con le GPU: installa il programma se manca, crea le riprese della
persona di prova sul set, ricostruisce la scena intera con un training lungo,
misura il risultato e prepara player e video. Dura qualche ora.

Fa più scene, una dopo l'altra.

- `prova_rapida` controlla soltanto che tutto funzioni su quel computer.
  Dura pochi minuti e viene rifatta a ogni avvio.
- `rig12_b`, `rig12_logit`, `rig12_fermo` e `rig12_solido` sono le stesse
  riprese (12 camere su un arco di 220°, 1280×720) addestrate con opzioni
  diverse.
- `sala16` è la scena da mostrare: 12 camere sull'arco e 4 in alto,
  1920×1080, con più passi di training. Usa le opzioni della prova che è
  riuscita meglio sulle camere di verifica.
- `rig4_b` ha 4 camere su un arco di 150°.

Alla fine:

- `consegna\` contiene, per ogni scena, `player.html` e `video.mp4`;
- `risultati.zip` contiene misure, immagini di controllo e registri. È il
  file da rimandare a chi sviluppa. Viene riscritto dopo ogni scena, quindi
  si può mandare anche prima che tutto sia finito.

Una scena già misurata viene saltata. Se si interrompe, riavviandolo riparte
da dove era arrivato. Da terminale:

```bash
python scripts/scena_di_prova.py --only sala16
```

Prima prova sulla workstation (due RTX 6000 Ada, 1280×720, 60 frame, scene
`rig4` e `rig12`): circa 24 passi al secondo, 14 minuti per i 20 000 passi
del set e 18 per i 25 000 della ripresa.

| Scena | Verifica: PSNR / PSNR soggetto / SSIM | Training: PSNR / PSNR soggetto |
|---|---|---|
| `rig12`, 12 camere | 24,6 dB / 21,9 dB / 0,86 | 34,0 dB / 26,6 dB |
| `rig4`, 4 camere | 16,7 dB / 14,4 dB / 0,67 | 30,4 dB / 26,5 dB |

Con 12 camere la persona regge anche vista da camere mai usate. Con 4 camere
no. In tutte e due il set era rovinato: semitrasparente, con il pavimento
sfocato. La causa era il decadimento dell'opacità, che nel set agiva a ogni
passo su tutte le Gaussiane e ne aveva tolte circa tre su quattro. Ora nel
set è spento (`scenery_opacity_decay`) e nella ripresa tocca solo le
Gaussiane che una camera ha disegnato in quel passo. In più, senza maschere
la scena viene addestrata su un fondo di colore casuale, che la obbliga a
essere piena. Le scene nuove servono a misurare l'effetto.

## Training sulle GPU

`train` usa tutte le GPU visibili (`devices: auto`). A ogni passo ogni GPU
disegna un istante diverso della clip, visto da `views_per_step` camere, e i
gradienti di tutte finiscono in un solo passo di ottimizzazione. Per usare una
sola GPU:

```bash
mister4d train data/take01 --out runs/take01 --devices cuda:0
```

I frame vengono decodificati in background e tenuti in RAM (`cache_gb`, 16 GB
di default). Se la clip non entra in memoria si riduce la risoluzione con
`prepare --downscale`.

Il numero di passi è automatico: 250 per frame della clip, almeno 7000.

## Come è rappresentato il 4D

Ogni Gaussiana ha, oltre agli attributi 3D, un istante, una durata e una
velocità. Nasce dal visual hull del suo frame, vive pochi frame e si
sovrappone a quelle degli istanti vicini. Tutti i frame sono ottimizzati
insieme. I dettagli sono in [docs/PIPELINE.md](docs/PIPELINE.md).

## Limiti

- Con 3–4 camere la ricostruzione regge dai punti di vista vicini a quelli
  ripresi. Il retro e le zone mai viste non vengono inventati, e `render`
  percorre solo l'arco coperto dalle camere. Tra una camera e l'altra il
  soggetto appare più largo del vero. `calibrate` e `train` dicono di quanto.
- `ingest` allinea le camere al frame più vicino. Lo scarto che resta si può
  far stimare al training (`--set learn_time_offsets=true`), ma con gesti
  rapidi è meglio la sincronizzazione hardware.
- Camere con esposizione o bianco diversi danno una scena che cambia tinta con
  il punto di vista. Il training può stimare un guadagno per camera
  (`--set learn_camera_colors=true`), ma serve un training lungo e non è stato
  provato su camere vere. È meglio bloccare esposizione e bianco su tutte.
- Le maschere dipendono dal set vuoto. Se luce o inquadratura cambiano tra le
  due riprese vengono male, e le ombre nette a terra possono entrarci. Si
  possono usare maschere fatte con altri strumenti.
- Le Gaussiane cambiano da un istante all'altro. Lontano dalle camere può
  restare un po' di sfarfallio, e non c'è una traiettoria per punto da usare
  per l'analisi del gesto.
- Tempo di calcolo, spazio su disco e peso del player crescono con la durata
  della ripresa. Tempi e peso su riprese vere sono ancora da misurare.
- Nel player il colore non cambia con il punto di vista, né nel browser né in
  Unity.
- Il risultato è fatto per essere guardato e non per prendere misure.

## Struttura

```
MISTER 4D Studio.bat   avvio di Studio con doppio clic (Windows)
Scena di prova su GPU.bat   la scena di prova intera, dal nulla al player (Windows)
LICENSE           licenza
unity/            componente per Unity (Quest 3) e descrizione del formato
src/mister4d/     pacchetto Python, un modulo per passo
  human.py        persona di prova per le riprese sintetiche (dati in assets/)
  scenery.py      il set vuoto: immagini di riferimento e punti di partenza
  run.py          lettura di una ripresa ricostruita, intera o a spezzoni
  lighten.py      copia leggera per visori autonomi
  launch.py       avvio di Studio: librerie, istanza unica, versione
  app.py          MISTER 4D Studio: server locale che guida i passi
  studio/         pagina di MISTER 4D Studio
  viewer/         player WebGL2 e colori (theme.css)
configs/          opzioni di training
scripts/          installazione (Windows e Linux), avvio di Studio (Linux), strumenti di misura
tests/            test automatici
docs/             come funziona, licenze, pipeline, sessione di ripresa
```

Test:

```bash
pytest
```

## Prossimi passi

Le lettere sono le fasi della guida operativa.

- B. Provare gsplat sulla workstation (`doctor`, `demo --run` a piena
  risoluzione) e misurare tempi e memoria con una e due GPU.
- C–D. Prima ripresa vera con due camere: pannello, set vuoto, oggetto fermo.
  Controllare `ingest`, sincronizzazione ed errore di calibrazione.
- E. Una breve azione con quattro camere, per tarare soglie delle maschere,
  densificazione e numero di passi.
- F–G. Con otto o più camere: `sh_degree` maggiore di zero, una camera di
  verifica dedicata, durate più lunghe per le parti ferme. Su GPU vanno
  provate anche due idee rimaste fuori, descritte in
  [docs/PIPELINE.md](docs/PIPELINE.md).
- H. Colore dipendente dalla vista nel player e prova del lettore Unity su
  Quest 3.

Il registratore con controllo delle camere resta fuori da questo progetto,
che parte dai file.
