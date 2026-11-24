# Come funziona MISTER 4D

Che cosa fa il programma e che cosa succede quando lo si usa, senza termini
tecnici.

## A che cosa serve

Trasforma la ripresa di una persona, fatta con alcune videocamere ferme messe
intorno a lei, in una scena che si può guardare da qualunque lato mentre si
muove. Mentre la scena scorre ci si può girare intorno, avvicinarsi e
allontanarsi.

## L'idea

Ogni camera vede la persona da un lato solo. Il programma mette insieme
quello che vedono tutte le camere nello stesso istante e ne ricava la forma e
i colori della persona nello spazio. Lo fa per ogni istante della ripresa. Il
risultato è una specie di scultura colorata che cambia nel tempo.

La scultura è fatta di tantissime piccole macchie di colore sfumate, sospese
nello spazio. Viste tutte insieme ricompongono l'immagine della persona. Le
macchie si chiamano gaussiane, e da lì viene il nome della tecnica, Gaussian
Splatting. 4D vuol dire tre dimensioni più il tempo.

## Che cosa serve

- Almeno tre videocamere ferme su cavalletti intorno al soggetto. Con più
  camere, ben distribuite, il risultato migliora.
- Un pannello stampato con un disegno a scacchi. Lo prepara il programma.
- Un computer con una o più schede grafiche NVIDIA.

## Che cosa si registra

Tre riprese, senza spostare le camere tra una e l'altra.

1. Il pannello, mosso dentro lo spazio di ripresa. Serve al programma per
   capire dove si trova ogni camera e dove guarda.
2. Lo spazio vuoto, per qualche secondo. Serve per separare la persona da
   tutto il resto.
3. La persona che fa quello che si vuole ricostruire.

## Che cosa fa il programma

Il lavoro è diviso in sei passi, proposti uno dopo l'altro.

1. Progetto. Si sceglie la cartella dove salvare il lavoro.
2. Video. Il programma controlla i video e li mette in pari, perché le camere
   non partono mai nello stesso momento. Per farlo cerca un segnale comune,
   per esempio un battito di mani.
3. Calibrazione. Guardando il pannello nei video, il programma calcola la
   posizione di ogni camera.
4. Preparazione. Il programma ritaglia la persona in ogni immagine
   confrontandola con lo spazio vuoto, e mostra un'anteprima da controllare.
5. Ricostruzione. Il programma parte da una forma grezza e la corregge
   moltissime volte, confrontandola ogni volta con le immagini vere. Può
   durare da qualche minuto a molte ore. Una ripresa lunga viene lavorata a
   pezzi di due secondi. Se il lavoro viene interrotto, alla ripartenza i
   pezzi già fatti restano.
6. Risultato. Il programma prepara il player, da aprire nel browser, e un
   video dimostrativo. Se serve prepara anche il materiale per chi deve
   montare un video o usare un visore.

Dopo ogni passo il programma dice com'è andata e, se qualcosa non va, che
cosa conviene rifare. Accanto a ogni richiesta c'è una piccola "i" che apre
la spiegazione. La voce "Come funziona questo passo" spiega il resto.

## Che cosa si ottiene

Il player. Per una ripresa breve è un solo file, player.html. Per una ripresa
lunga è una cartella che contiene index.html, da consegnare intera. Si apre
con doppio clic, anche su un altro computer e senza connessione.

Se serve si ottengono anche:

- un filmato del solo soggetto con lo sfondo trasparente, da usare in un
  programma di montaggio;
- una versione più leggera della scena, per i visori che funzionano senza
  computer;
- la scena in un formato standard, per altri programmi di grafica 3D.

## Come la scena resta pulita

Con poche camere, alcune macchie finiscono in punti dove non c'è niente.
Dalle camere non si notano, ma spostandosi in un punto dove non c'era
nessuna camera si vedono come aloni.

Per toglierle, a ogni correzione ogni macchia usata perde un po' della sua
forza. Quelle che servono a ricomporre le immagini la riprendono subito. Le
altre continuano a perderla, svaniscono e vengono tolte. Nelle prove la scena
è risultata un po' più fedele vista da un punto nuovo, e con meno macchie.

Il programma impedisce anche che una macchia diventi un ago sottile o più
piccola di quello che le camere riescono a distinguere. Macchie così non si
notano dalle camere, ma ingrandendo la scena si vedono come peli e strisce
attorno al soggetto.

## Anche il set, non solo la persona

Di norma il programma ritaglia la persona e lascia fuori tutto il resto. Si
può chiedere invece di ricostruire la scena intera. In quel caso il programma
ricostruisce prima il set vuoto, che non si muove, e poi ci mette sopra
quello che si muove. Se il set vuoto non è stato ripreso lo ricava dalla
ripresa stessa, guardando che cosa resta fermo. Per ora questa scelta si fa
solo da terminale.

## Quanto pesa la scena

Si può scegliere tra tre pesi: massimo dettaglio, bilanciata, leggera. Nella
scena leggera ogni macchia dura più a lungo e segue il movimento del
soggetto, e ne serve circa un terzo. È la scelta giusta per le riprese lunghe
e per i visori.

## Da che cosa dipende la qualità

- Dal numero di camere e da come sono disposte. Dove nessuna camera guarda,
  il programma può solo tirare a indovinare.
- Dalla sincronizzazione. Se le camere non riprendono lo stesso istante, i
  movimenti veloci vengono confusi.
- Dalla luce, che deve essere uniforme e non cambiare durante le riprese.
- Dai colori. Camere diverse, o lasciate in automatico, vedono lo stesso
  soggetto con colori un po' diversi, e la scena cambia tinta quando ci si
  sposta. Conviene bloccare esposizione e bianco su tutte. Se non basta, nel
  passo Ricostruzione c'è una casella che fa stimare e correggere la
  differenza.
- Dalla precisione della calibrazione.

## Che cosa non fa

- Non funziona in diretta. Prima si registra, poi si elabora.
- Non inventa quello che nessuna camera ha visto.
- Non è veloce. Il tempo di calcolo cresce con la durata della ripresa.
- Il risultato è fatto per essere guardato e non per prendere misure.

## Avvio e aggiornamenti

Il programma si apre con doppio clic su "MISTER 4D Studio". A ogni avvio
controlla tre cose.

- Che ci siano tutte le librerie che gli servono. Se un aggiornamento ne ha
  aggiunta una, la installa senza toccare le altre.
- Che non sia già aperto. In quel caso mostra la finestra già aperta.
- Che la pagina e il programma che lavora dietro siano della stessa versione.

Se il programma viene aggiornato mentre è aperto, la pagina chiede di
chiuderlo e riaprirlo, e fino ad allora non fa partire nuovi passi. Se viene
riaperto lasciando aperta la pagina di prima, la pagina si aggiorna da sola.

## Dove restano i dati

Tutto avviene sul computer su cui il programma è installato. Nessun video e
nessun risultato viene mandato altrove.
