# Sessione di ripresa: che cosa registrare per questo software

Traduce le indicazioni della *Guida operativa set 4DGS* in ciò che la pipeline
si aspetta in ingresso. Per ogni sessione servono **tre clip**, ciascuna con
**un file video per camera**.

```
sessione/
  raw/
    calib/   cam01.mp4  cam02.mp4  cam03.mp4      pannello ChArUco in movimento
    empty/   cam01.mp4  cam02.mp4  cam03.mp4      set vuoto
    take01/  cam01.mp4  cam02.mp4  cam03.mp4      soggetto
```

Regole sui file:

- il **nome del file è il nome della camera** e deve essere identico nelle tre
  clip (`cam01.mp4` è sempre la stessa camera fisica);
- stessa risoluzione e stesso frame rate per tutte le camere;
- formati letti da FFmpeg (MP4/MOV H.264, H.265, ProRes, …). Il **Blackmagic
  RAW (`.braw`) non è leggibile da FFmpeg**: va prima convertito, per esempio
  con DaVinci Resolve, mantenendo risoluzione e frame rate;
- gli originali non vengono mai modificati: `ingest` li legge, ne registra il
  checksum SHA-256 e scrive i frame estratti altrove.

## Prima di registrare

- Camere **fisse**: dopo la calibrazione non si toccano posizione, zoom, fuoco.
  Se succede, si ripete la clip `calib`.
- Fuoco manuale, esposizione e bilanciamento del bianco **bloccati**, uguali
  su tutte le camere. Tempo di esposizione breve (punto di partenza 1/125 –
  1/250 s) per limitare il mosso.
- Luce ampia, diffusa e stabile; fondale e pavimento opachi. Le maschere si
  ottengono per confronto con il set vuoto: ogni variazione di luce o di
  inquadratura tra `empty` e `take` finisce nella maschera.
- Niente stabilizzazione, niente inseguimento automatico del soggetto (PTZ),
  niente ritaglio digitale variabile.

## 1. Clip `calib`: pannello ChArUco

Stampa del pannello:

```bash
mister4d board --out charuco_5x7.png
```

Di default: 5×7 riquadri da 12 cm (60×84 cm, circa un A1), marker da 9 cm,
dizionario `DICT_4X4_50`. Stampare al 100% su un supporto **rigido e opaco**,
poi **misurare un riquadro** con un calibro: il valore reale va passato a
`calibrate --square` (e `--marker`), perché è il pannello a fissare la scala
metrica di tutto il rig.

Come muoverlo:

- attraversare tutto il volume, a varie altezze e distanze, inclinando il
  pannello in direzioni diverse; portarlo anche verso i bordi delle
  inquadrature (il comando segnala se la copertura dell'immagine è scarsa);
- il pannello deve essere visto **nello stesso istante da camere adiacenti**:
  è così che le camere vengono collegate tra loro. Con camere tutto intorno,
  ruotare il pannello passo passo lungo il cerchio. Se una camera non viene mai
  collegata alle altre, `calibrate` si ferma dicendo quale;
- 30–60 secondi bastano. Con camere **non** sincronizzate via hardware,
  muovere il pannello lentamente o a scatti (fermo – sposta – fermo): un
  errore di sincronizzazione mentre il pannello si muove diventa errore di
  calibrazione;
- facoltativo ma consigliato: per un istante appoggiare il pannello **a terra
  al centro del volume**, lato stampato verso l'alto. Indicando quel frame con
  `calibrate --floor-frame N` l'origine va al centro del pannello e il
  pavimento a z = 0. Senza, il sistema mondo è ricavato dal rig (origine nel
  punto verso cui guardano le camere, verticale dedotta dall'orientamento
  delle camere, quota del pavimento ignota).

## 2. Clip `empty`: set vuoto

Pochi secondi, senza soggetto e senza operatori in campo, con **le stesse
impostazioni e la stessa luce** della ripresa. Va rifatta se cambia qualcosa
sul set.

## 3. Clip `take`: il soggetto

- Il soggetto resta dentro il volume inquadrato da tutte le camere.
- Per il primo collaudo seguire la scaletta della guida: soggetto fermo →
  movimento lento → gesto con mani e piccola rotazione. Clip brevi (5–10 s).
- Abiti opachi e con trama aiutano; evitare colori simili al fondale.

## Sincronizzazione

Il software **non può rendere simultanee esposizioni che non lo erano**: può
solo allineare i file al frame più vicino e dire quanto errore resta.

| Situazione | Opzione di `ingest` | Che cosa fa |
|---|---|---|
| Camere con trigger/genlock, registrazione avviata insieme | `--sync none` | Nessun allineamento: usa i timestamp dei file. |
| Avvio registrazione non simultaneo, audio presente | `--sync audio` | Allinea sul sonoro (un battito di mani all'inizio di ogni clip). Riporta l'errore residuo sotto il frame. |
| Nessun audio | `--sync flash` | Allinea sul frame con il maggior salto di luminosità (flash o luce accesa di colpo, visibile a tutte le camere). |
| Offset già noti | `--offsets cam02=3,cam03=-1` | Offset manuali in frame. |

Dopo l'evento di sync usare `--start N` per scartare i primi frame (battito di
mani, flash, ingresso del soggetto).

L'errore che resta sotto il frame si può recuperare in fase di training con
`--set learn_time_offsets=true`: la scena è continua nel tempo e il training
stima per ogni camera di quanto ha esposto prima o dopo la camera di
riferimento. Non sostituisce il sync hardware (lo scarto viene stimato, non
misurato, e serve che il soggetto si muova), ma riduce l'errore con camere che
ne sono prive.

Che cosa controllare nell'output di `ingest` (e in `ingest.json`):

- `dropped frame`: un frame mancante **non viene rinumerato**; quell'istante
  viene escluso per tutte le camere e segnalato;
- `sub-frame sync error`: errore residuo non correggibile. Ordine di
  grandezza: una mano a 1 m/s percorre 1 cm in 10 ms; a 30 fps l'errore può
  arrivare a ±16 ms. Per gesti rapidi serve la sincronizzazione hardware;
- `weak sync peak`: l'evento di sync non è chiaro, verificare a occhio;
- frame rate o risoluzione diversi tra camere.

## Dove mettere le camere

Conta come sono distribuite le **linee di vista**, più del numero: ogni camera
vincola la forma solo di traverso alla propria direzione, e due camere una di
fronte all'altra vedono la stessa sagoma. Dopo la calibrazione, `calibrate`
stampa l'angolo più grande rimasto scoperto e di quanto il soggetto può
apparire più largo guardandolo da lì. Regole pratiche:

- distribuire le linee di vista su 180°, evitando coppie esattamente opposte
  (8 camere ogni 45° valgono 4 linee di vista; 9 ogni 40° ne valgono 9);
- con poche camere, tenerle vicine tra loro attorno al lato che interessa:
  3 camere a 30–40° l'una dall'altra coprono bene un settore, 3 camere a 90°
  lasciano forme molto incerte;
- alzare e sfalsare alcune camere aggiunge linee di vista diverse.

## Quante camere

| Camere | Che cosa aspettarsi da questo POC |
|---|---|
| 2 | Solo verifica di acquisizione, sync e calibrazione. Il training parte ma il risultato non è significativo. |
| 3–4 | Ricostruzione di un **settore**: buona dai punti di vista vicini alle camere, sempre più incerta allontanandosi. Il retro e le zone mai viste non vengono inventate. |
| 8+ in cerchio | Copertura a 360°; si può alzare `sh_degree` e tenere una camera fuori dal training per misurare la qualità. |

Per misurare la qualità serve un punto di vista indipendente: `train --holdout
camXX` esclude una camera dal training e `eval` la usa come verità. Con 3
camere l'esclusione ne lascia 2 al training: il numero che si ottiene descrive
quella configurazione ridotta, non il rig completo.

## Scheda della sessione (da compilare)

- data, operatori, soggetto, liberatoria
- modello camere, firmware, ottica, focale, diaframma, tempo, ISO, WB, fps
- tipo di sincronizzazione e riferimento usato
- dimensioni misurate del riquadro del pannello
- note: luci, fondale, anomalie, clip da scartare
