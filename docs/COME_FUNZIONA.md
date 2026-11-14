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
   costruito con le immagini vere, finché le due cose si assomigliano.
   Intanto toglie le macchie che non servono (vedi più sotto). Può durare da
   qualche minuto a molte ore. Una ripresa lunga viene lavorata a
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

## Come il programma tiene pulita la scena

Con poche camere ci sono molti modi di disporre le macchie che danno le stesse
immagini viste dalle camere, ma non tutti reggono quando ci si sposta in un
punto in cui nessuna camera c'era: alcune macchie stanno dove non c'è niente
e da lì si vedono come aloni.

Per questo, a ogni correzione, ogni macchia che è stata usata perde un po'
della sua forza. Quelle che servono davvero a ricomporre le immagini la
riprendono subito; quelle che stanno lì senza motivo continuano a perderla,
svaniscono e vengono tolte. Nelle prove la scena è risultata un po' più
fedele vista da un punto nuovo, con meno macchie.

L'idea viene da ricerche pubblicate nel 2025 e nel 2026 sulla ricostruzione
con poche camere. Insieme a questa ne sono state provate altre tre: misurate,
non hanno dato un miglioramento e sono rimaste fuori. Il programma che applica
queste idee è stato scritto qui, senza copiare codice altrui.

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
- Dai colori. Camere diverse, o lasciate in automatico, vedono lo stesso
  soggetto con colori un po' diversi, e la scena cambia tinta quando ci si
  sposta. Conviene bloccare esposizione e bianco su tutte. Se non basta, nel
  passo Ricostruzione c'è una casella che fa stimare al programma la
  differenza di ogni camera e la compensa.
- Dalla precisione della calibrazione.

## Che cosa non fa

- Non funziona in diretta: prima si registra, poi si elabora.
- Non inventa quello che nessuna camera ha visto.
- Non è veloce: il tempo di calcolo cresce con la durata della ripresa.
- Il risultato è fatto per essere guardato, non per prendere misure.

## Avvio e aggiornamenti

Il programma si apre con doppio clic su "MISTER 4D Studio". A ogni avvio
controlla da solo tre cose.

- Che ci siano tutte le librerie che gli servono. Se un aggiornamento ne ha
  aggiunta una, la installa, senza toccare le altre.
- Che non sia già aperto. In quel caso mostra la finestra già aperta invece di
  aprirne una seconda.
- Che la pagina che si vede e il programma che lavora dietro siano della
  stessa versione.

Se il programma viene aggiornato mentre è aperto, la pagina lo dice e chiede
di chiuderlo e riaprirlo; fino ad allora non fa partire nuovi passi. Se viene
riaperto lasciando aperta la pagina di prima, la pagina si aggiorna da sola.

## Dove restano i dati

Tutto avviene sul computer su cui il programma è installato. Nessun video e
nessun risultato viene mandato altrove.
