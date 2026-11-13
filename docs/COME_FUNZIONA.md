# Come funziona MISTER 4D

Questo documento spiega che cosa fa il programma e che cosa succede quando lo
si usa, senza termini tecnici.

## A che cosa serve

Serve a trasformare la ripresa di una persona, fatta con alcune videocamere
ferme messe intorno a lei, in una scena che si può guardare da qualunque punto
di vista mentre si muove. Non è un video normale: mentre la scena scorre ci si
può girare intorno, avvicinarsi e allontanarsi.

## L'idea

Ogni camera vede la persona da un lato solo. Il programma mette insieme quello
che vedono tutte le camere nello stesso istante e ne ricava la forma e i
colori della persona nello spazio. Lo fa per ogni istante della ripresa. Il
risultato è come una scultura colorata che cambia nel tempo.

La scultura non è fatta di superfici ma di tantissime piccole macchie di
colore sfumate, sospese nello spazio. Viste tutte insieme, da qualunque lato,
ricompongono l'immagine della persona. Queste macchie si chiamano gaussiane, e
da qui viene il nome della tecnica: Gaussian Splatting. "4D" vuol dire tre
dimensioni più il tempo.

## Che cosa serve

- Almeno tre videocamere, meglio di più, ferme su cavalletti intorno al
  soggetto. Più camere ci sono e meglio sono distribuite, migliore è il
  risultato.
- Un pannello stampato con un disegno a scacchi, che il programma stesso
  prepara.
- Un computer con una o più schede grafiche NVIDIA.

## Che cosa si registra

Tre riprese, senza mai spostare le camere tra l'una e l'altra.

1. Il pannello, mosso dentro lo spazio di ripresa. Serve al programma per
   capire dove si trova ogni camera e in che direzione guarda.
2. Lo spazio vuoto, per qualche secondo. Serve per distinguere poi la persona
   da tutto il resto.
3. La persona che fa quello che si vuole ricostruire.

## Che cosa fa il programma

Il lavoro è diviso in sei passi, che il programma propone uno dopo l'altro.

1. Progetto. Si sceglie la cartella dove salvare il lavoro.
2. Video. Il programma controlla i video e li mette in pari, perché le camere
   non partono mai tutte nello stesso momento. Per farlo cerca un segnale
   comune, per esempio un battito di mani.
3. Calibrazione. Guardando il pannello nei video, il programma calcola la
   posizione esatta di ogni camera. Senza questo passo non potrebbe combinare
   le immagini.
4. Preparazione. Il programma ritaglia la persona in ogni immagine,
   confrontandola con lo spazio vuoto, e mostra un'anteprima da controllare.
5. Ricostruzione. È il passo lungo. Il programma parte da una forma grezza e
   la corregge moltissime volte, confrontando ogni volta quello che ha
   costruito con le immagini vere, finché le due cose si assomigliano. Può
   durare da qualche minuto a molte ore. Una ripresa lunga viene lavorata a
   pezzi di due secondi, uno dopo l'altro: se il lavoro viene interrotto, alla
   ripartenza i pezzi già fatti restano.
6. Risultato. Il programma prepara il player, da aprire nel browser, in cui si
   guarda la scena e ci si gira intorno, e un video dimostrativo. Su richiesta
   prepara anche il materiale per chi deve montare un video o usare un visore.

Dopo ogni passo il programma dice com'è andata e, se qualcosa non va, che cosa
conviene rifare. Accanto a ogni richiesta c'è una piccola "i": premendola
compare la spiegazione di quella richiesta. La voce "Come funziona questo
passo" racconta il resto.

## Che cosa si ottiene

Il player. Per una ripresa breve è un solo file, player.html; per una ripresa
lunga è una cartella, da consegnare intera, che contiene index.html. In
entrambi i casi si apre con doppio clic, anche su un altro computer e senza
connessione. Si può far partire e fermare la scena e guardarla da dove si
vuole.

Su richiesta si ottengono anche:

- un filmato del solo soggetto con lo sfondo trasparente, da sovrapporre ad
  altre immagini in un programma di montaggio;
- una versione più leggera della scena, pensata per i visori che funzionano da
  soli, senza computer;
- la scena in un formato standard, per altri programmi di grafica 3D.

## Quanto pesa la scena

Si può scegliere tra tre pesi: massimo dettaglio, bilanciata, leggera. Nella
scena leggera ogni macchia di colore dura più a lungo e segue il movimento del
soggetto, invece di essere sostituita a ogni istante: ne servono circa un
terzo. È la scelta adatta alle riprese lunghe e ai visori.

## Da che cosa dipende la qualità

- Dal numero di camere e da come sono disposte. Dove nessuna camera guarda, il
  programma può solo indovinare.
- Dalla sincronizzazione. Se le camere non riprendono lo stesso istante, i
  movimenti veloci vengono confusi.
- Dalla luce, che deve essere uniforme e non cambiare durante le riprese.
- Dalla precisione della calibrazione.

## Che cosa non fa

- Non funziona in diretta: prima si registra, poi si elabora.
- Non inventa quello che nessuna camera ha visto.
- Non è veloce: il tempo di calcolo cresce con la durata della ripresa.
- Il risultato è fatto per essere guardato, non per prendere misure.

## Dove restano i dati

Tutto avviene sul computer su cui il programma è installato. Nessun video e
nessun risultato viene mandato altrove.
