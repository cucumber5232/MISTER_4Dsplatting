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
   durare da qualche minuto a qualche ora.
6. Risultato. Il programma prepara un file da aprire nel browser, in cui si
   guarda la scena e ci si gira intorno, e un breve video dimostrativo.

Dopo ogni passo il programma dice com'è andata e, se qualcosa non va, che cosa
conviene rifare. In ogni passo la voce "Che cosa viene chiesto, in dettaglio"
spiega meglio ogni richiesta.

## Che cosa si ottiene

Un file chiamato player.html. Si apre con doppio clic, anche su un altro
computer e senza connessione. Si può far partire e fermare la scena e guardarla
da dove si vuole.

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
- È pensato per riprese brevi, di pochi secondi.
- Il risultato è fatto per essere guardato, non per prendere misure.

## Dove restano i dati

Tutto avviene sul computer su cui il programma è installato. Nessun video e
nessun risultato viene mandato altrove.
