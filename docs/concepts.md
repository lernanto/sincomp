# Concepts and terminology

This chapter introduces the linguistic and machine-learning concepts used
in the SinComp documentation.

## Phonetics and phonology

**Phonetics** is the study of speech production, transmission, and perception.
It describes physical or perceptual units such as duration, fundamental
frequency, formants, place and manner of articulation, and consonant or
vowel quality. **Phonology** studies how a language organizes and contrasts
the available sounds. It asks, for example, whether two sounds can occur
in the same context and distinguish words.[^ipa][^catford]

An utterance can be divided into a sequence of segments. A **segment**
is a unit represented as one position in a sequential speech
representation.

A **phone** is a speech sound considered phonetically, without
automatically assigning it phonemic status.

Phones are classified as consonants or vowels. A **consonant** is
produced with a constriction or complete closure of the vocal tract. A
**vowel** is produced with a relatively open vocal tract and usually a
sonorant nucleus.

Consonants can be classified by:

- **place of articulation**, such as labial or alveolar;
- **manner of articulation**, such as stop, nasal, or fricative;
- **laryngeal setting**, such as voiced or voiceless.

Vowels can be classified by tongue height, tongue backness, roundedness,
and other features.

A **phoneme** is a language-specific category; different phones can be
allophones of one phoneme, while two phonemes contrast in minimal pairs.
The distinction is not merely descriptive: a phoneme is an abstraction
about a contrastive system.[^crystal]

Phonological **features** represent properties used to classify or
contrast segments. A **distinctive feature** is a feature relevant to
maintaining a contrast. Features may also form larger patterns, such as
voice, nasality, or tone.

**Syllable structure** organizes segments above the segment level. A
syllable typically has an onset and a nucleus, and may have a coda.

```text
syllable = onset + nucleus + (coda)
```

**Suprasegmental features** extend across a syllable or larger domain.
**Tone** is a suprasegmental pitch pattern associated with a syllable or
morpheme. A **tonal language** uses pitch categories
systematically in lexical or grammatical distinctions. Intonation,
stress, and length can
also be suprasegmental; they are not the same as tone.

**Allophonic variation** occurs when context changes the realization of
a phoneme. Contextual variants are **conditioned variants**, or
**allophones**. An **unconditioned variant** is not accounted for by a
simple phonetic context, though sociolinguistic or lexical factors may
explain it.

The following terms concern the descriptive levels used throughout this
guide:

| Term | Definition |
| --- | --- |
| Segment | A sequenceable unit of speech, represented separately from neighboring units. |
| Phone | A phonetically describable speech sound, not necessarily a phoneme. |
| Consonant | A phone produced with a constriction or closure of the vocal tract. |
| Vowel | A phone produced with a relatively open vocal tract and a sonorant nucleus. |
| Phoneme | A category in a contrastive, language-specific sound system. |
| Allophone | A contextually distributed realization of a phoneme. |
| Conditioned variant | A variant governed by a phonetic or prosodic context. |
| Unconditioned variant | A variant not explained by a simple local phonetic context. |
| Feature | A property used to classify, contrast, or transform segments or tones. |
| Distinctive feature | A feature relevant to a phonological contrast. |
| Syllable | A prosodic unit, commonly containing a nucleus and possibly an onset and coda. |
| Onset | The segment or segments before the syllable nucleus. |
| Nucleus | The central, usually vowel, portion of a syllable. |
| Coda | The segment or closure after the syllable nucleus. |
| Tone | A pitch pattern or pitch category associated with a syllable or morpheme. |
| Suprasegmental | A property extending over a segment, syllable, word, or larger unit. |

## Historical linguistics

**Historical linguistics** studies how languages change and how related
languages descend from earlier stages. A **proto-language** is a
reconstructed system inferred from present and earlier evidence; it is
not a directly recorded speech variety.

A reconstruction usually begins with systematic correspondences
between forms in related languages or dialects. The **comparative
method** evaluates these correspondences, identifies cognates, and
uses regular sound change and relative chronology to infer earlier
forms. Borrowing, onomatopoeic resemblance, analogy, and chance
similarity must also be considered.[^campbell][^crowley-bowern]

A **sound change** describes a recurrent change in a sound or sound
class under specified conditions. A **correspondence** is a relationship
between units in related varieties; it is not by itself a proof of
identity or direct inheritance. **Shared innovation** refers to a change
found in a proposed subgroup and absent from the reconstructed ancestor.

**Proto-forms** are reconstructed expressions. A **cognate** is a form
inherited from a common ancestor. **Analogy** generalizes a pattern and
can replace a regular correspondence. A **family tree** represents
branching descent, while **reticulation** or a network represents
borrowings and convergences that a simple tree does not capture.[^nakhleh]

Genealogical classifications use several layers, including language
family / phylum, branch, language group / subgroup, and language.
Authorities differ in their use of *stock*, *family*,
*branch*, and *subgroup*. In one common classification, Sino-Tibetan
is the language family, Sinitic is the Chinese branch or subgroup,
and Sinitic languages are the descendant varieties compared by
Chinese historical linguistics.[^glottolog]

| Term | Definition |
| --- | --- |
| Historical linguistics | The study of language change and language history. |
| Comparative linguistics | The comparison of likely related varieties or reconstructed stages. |
| Comparative method | A procedure for inferring regular correspondence and proto-forms. |
| Sound change | A recurrent change in a sound or sound class. |
| Sound correspondence | A recurrent relationship between corresponding units in related varieties. |
| Shared innovation | A change supporting membership in a proposed historical subgroup. |
| Borrowing | Transfer of a form or pattern through language contact. |
| Proto-language | A reconstructed ancestor of a language or dialect group. |
| Proto-form | A reconstructed form in a proto-language or proto-system. |
| Reconstruction | The inference of earlier forms or systems from comparative evidence. |
| Cognate | A form inherited from a common ancestor. |
| Relative chronology | The ordering of changes without absolute dates. |
| Family tree | A branching model of genealogical descent. |
| Reticulation / network | A model with borrowing or convergence between branches. |

## Dialectology

In general linguistics, language, dialect, variety, and accent are partly
overlapping concepts. A **language** is a
structured communicative system or a socially and politically
recognized grouping of varieties. A **variety** is a neutral term for a
distinguishable form of language. A **dialect** is a variety associated
with a region or social group and differing in pronunciation, grammar,
or vocabulary. An **accent** usually emphasizes pronunciation
differences relative to a reference variety.

The terms are not interchangeable with every meaning of *language*:
mutual intelligibility, ethnicity, government policy, and social identity
can give different answers for the same pair of varieties. "Dialect"
does not mean that a variety is subordinate, ungrammatical, or without
a literature.[^chambers-trudgill]

There are also broad social distinctions:

- a **regional dialect** varies primarily by region;
- a **social dialect**, or sociolect, varies primarily by social group;
- a **register** varies by situation or style rather than population;
- an **idiolect** is an individual's systematic linguistic patterns.

**Dialectology** combines dialect geography, historical linguistics,
sociolinguistics, and phonetics. It examines how varieties are
distributed, how they change, and how features differ among places and
communities.

**Dialect geography**, or linguistic geography, maps and compares the
distribution of linguistic features. An **isogloss** is a boundary-like
line separating an area with a feature or feature value from an area
without it. A **bundle of isoglosses** marks a concentration of
boundaries. A **transition zone** contains gradients or overlapping
varieties rather than a single sharply defined boundary.

A **dialect continuum** is a chain of neighboring varieties with
gradually changing differences. A **dialect atlas** records comparable
responses from a set of locations. An **areal feature** is shared by
neighboring varieties because of contact or diffusion; it should not
automatically be treated as inherited.[^dialect-atlas]

| Term | Definition |
| --- | --- |
| Language | A structured and socially recognized linguistic system or grouping. |
| Variety | A neutral term for a distinguishable form of language. |
| Dialect | A regional or social variety differing from a reference variety. |
| Accent | Pronunciation or prosodic differences associated with a speaker or group. |
| Regional dialect | A variety distinguished primarily by geographic distribution. |
| Social dialect / sociolect | A variety distinguished primarily by social group. |
| Register | A style or variety selected for a situation or communicative purpose. |
| Idiolect | An individual's characteristic linguistic system or pattern. |
| Dialect geography / linguistic geography | The study of geographic variation and distribution in language. |
| Dialect atlas | A collection of comparable linguistic maps. |
| Isogloss | A line marking the geographic distribution of a feature. |
| Bundle of isoglosses | Several isoglosses occurring in a similar geographic region. |
| Transition zone | A region with gradual or overlapping variation. |
| Dialect continuum | Neighboring varieties linked by gradual, continuous variation. |
| Areal feature | A feature shared through geographic spread or contact. |
| Areal convergence | Increased similarity among varieties in a region through contact. |
| Language border | A spatial or social boundary between strongly contrasted varieties. |

## IPA

The **International Phonetic Alphabet (IPA)** is a standardized set of
phonetic symbols for describing speech. It is intended to represent
physical speech units rather than a language-specific phoneme inventory.
An IPA symbol is therefore not automatically a phoneme, and a language
may use diacritics or a source-specific transcription for additional
detail.[^ipa]

The IPA distinction between slash brackets and square brackets is
important:

- `/p/` describes a phonemic unit or category in a stated analysis;
- `[pʰ]` describes an aspirated phonetic realization;
- `[p]` by itself does not identify whether a source means a consonant
  symbol, a phonetic category, or a dataset label.

Tone may be represented by a numeric pitch contour, a pitch class, or a
tone name. Chao-style numbers such as `55`, `35`, `31`, and `213`
describe a relative pitch pattern. A bare integer category such as `1`
or a source-specific `A`/`B` label is a phonological category, not a
physical pitch measurement.[^chao]

In Chinese dialectology, a **zero phoneme** is often represented
by the empty-set symbol, `∅`. A **zero segment** means that no
segment is realized in a position.

## Chinese dialects

Chinese is characterized by the predominance of **monosyllabic
morphemes** (单音节语素): most morphemes are monosyllabic, so
phonetic analysis usually begins by analyzing syllables.

All known varieties of Chinese have tone. A tonal
language uses pitch patterns systematically to distinguish
or add a contrast to words or other units. Tone is always associated
with a syllable, though it may not be lexically distinctive for every
morpheme or every syllable in a particular analysis.

Traditional Chinese phonological descriptions represent a syllable as:

```text
syllable
├── initial (optional)
├── final
│   ├── medial (optional)
│   ├── nucleus
│   └── coda (optional)
└── tone
```

For example, the character *光* is pronounced /kuɑŋ⁵⁵/ in *Beijing dialect* (北京话):

```text
syllable: kuɑŋ⁵⁵
├── initial: k
├── final: uɑŋ
│   ├── medial: u
│   ├── nucleus: ɑ
│   └── coda: ŋ
└── tone: ⁵⁵
```

An **initial** (声母) corresponds approximately to the onset of a
syllable. A **final** (韵母) includes the nucleus and possible coda and
may include a medial. The **nucleus** (韵腹) is the main vowel, not the
whole final. A **coda** (韵尾) is the segment or closure after the
nucleus. Tone is separate from the segmental structure, although source
encodings combine all three fields into a complete syllable string.

In Chinese dialectology, **dialect** (方言) names varieties grouped
within Chinese as one language, even when some of those varieties are
not mutually intelligible. The category can therefore include varieties
that might be classified as separate languages under other criteria.
Here, dialect usually refers to regional varieties and does not
include social dialects.

Chinese dialectology normally organizes varieties into a hierarchy of
**dialect groups** (方言区), **dialect clusters** (方言片),
**dialect sub clusters** (方言小片), and **dialect spots** (方言点):

```text
dialect group > dialect cluster > dialect sub cluster > dialect spot
```

The terms and boundaries vary among atlases. Examples of major dialect
groups include *Mandarin* (官话), *Wu* (吴语), *Min* (闽语), *Yue* (粤语),
*Hakka* (客家话), *Gan* (赣语), *Xiang* (湘语), and *Jin* (晋语).[^termonline]

Within some dialect groups, such as Mandarin and Min, there is also a division
between group and cluster, without formal level name. For example, one classification of *Jinan dialect* (济南话) can be represented
as:

```text
group: 官话
└── (unnamed): 冀鲁官话
    └── cluster: 石济片
        └── sub cluster: 聊泰小片
            └── spot: 济南话
```

Chinese dialectology is closely related to **Chinese historical
phonology** (音韵学). Dialect comparison is often interpreted
using historical rime tables and traditional rime categories such as
the *Qieyun* (切韻) rhyme dictionary. [^norman][^pulleyblank]

Chinese dialect records often preserve several chronological
**historical strata** (历史层次): layers of forms inherited or borrowed
at different periods. A dialect may retain readings that entered its
history at different times.

**Literary-colloquial reading difference** (文白异读) refers to a
character in a dialect having distinct literary and colloquial
pronunciations, often with different lexical or stylistic distributions.
A **literary reading** (文读) is associated with learned, formal, or
text-based vocabulary in that dialect; a **colloquial reading** (白读)
is associated with local vernacular speech.

Separately, **literary transmission** (文传) and
**colloquial transmission** (俗传) denote historical pathways by which forms
are passed through learned literary practices or local vernacular
speech, respectively.[^norman] These transmission categories are not
one-to-one counterparts of literary reading and colloquial reading: a reading's current literary
or colloquial use does not by itself determine its historical
transmission path.

| Term | Chinese | Definition |
| --- | --- | --- |
| Morpheme | 语素 | The smallest meaningful unit of a language. |
| Monosyllabic morpheme | 单音节语素 | A morpheme consisting of one syllable. |
| Initial | 声母 | The onset or initial consonant of a Chinese syllable. |
| Final | 韵母 | The part after the initial, including nucleus and possible coda and medial. |
| Medial | 韵头 | A glide-like element before the nucleus. |
| Nucleus | 韵腹 | The main vowel of a final. |
| Coda | 韵尾 | The segment or closure after the nucleus. |
| Chinese dialect | 汉语方言 | A regional variety categorized within Chinese. |
| Dialect group | 方言区 | A broad conventional grouping, for example Mandarin or Yue. |
| Dialect cluster | 方言片 | A division within a dialect group. |
| Dialect sub cluster | 方言小片 | A finer division within a dialect cluster. |
| Dialect spot | 方言点 | A surveyed locality or source-specific variety record. |
| Historical stratum | 历史层次 | A layer of forms inherited or borrowed at a different period. |
| Literary reading | 文读 | A reading associated with learned or formal usage in a dialect. |
| Colloquial reading | 白读 | A reading associated with local vernacular usage in a dialect. |
| Literary-colloquial reading difference | 文白异读 | Distinct literary and colloquial readings in a dialect; not a statement of their transmission history. |
| Literary transmission | 文传 | Historical transmission through learned literary practices. |
| Colloquial transmission | 俗传 | Historical transmission through local vernacular speech. |

## Chinese characters

Chinese historical phonology and dialectology have traditionally used
**Chinese characters** (汉字) as units of analysis. A Chinese character
typically corresponds to one **monosyllabic morpheme** (单音节语素).
The correspondence is not strict: **homography** (同形字) allows distinct
morphemes to share one **glyph** (字形), while, more rarely, a single
morpheme has **allographs** (异体字).[^unicode][^crystal]

Chinese text uses **traditional characters** (繁体字) or
**simplified characters** (简化字). Simplification can merge several
traditional forms under one simplified form, while leaving their
traditional forms distinct.[^unicode]

A **polyphone** (多音字) is an item with more than one reading within a
particular dialect or dialect spot.[^norman]

Polyphony can arise in several ways:

- historical polyphony retained in a dialect;
- lexical or phonological differentiation within a dialect;
- mixed inheritance from different dialect traditions;
- literary reading and colloquial reading;
- simplified-character mergers such as *后*, *干*, *只*, and other many-to-one
  mappings.

For historical polyphony retained in one dialect and for
simplified-character mergers, the readings can often be readily
identified as corresponding to different morphemes. In the other cases,
the relationship between readings and morphemes may be fuzzy.

| Term | Chinese | Definition |
| --- | --- | --- |
| Chinese character | 汉字 | A written unit in the Chinese writing system. |
| Traditional character | 繁体字 | A character preserved in a traditional orthography. |
| Simplified character | 简化字 | A character produced or selected under simplified orthography. |
| Polyphone | 多音字 | An item with more than one reading within a particular dialect or dialect spot. |
| Glyph | 字形 | A graphic form used to represent a character or symbol. |
| Homography | 同形 | Distinct morphemes sharing the same written form. |
| Allograph | 异体字 | An alternative written form of the same morpheme. |

## Machine learning

**Machine learning** uses data to estimate patterns or
decision rules, rather than requiring an expert to specify every rule
that maps an input to an output. An **expert system**
typically applies knowledge explicitly encoded by people as rules and
facts. Machine learning can still use expert-designed representations,
constraints, and rules; the distinction is that at least some mapping or
parameters are inferred from examples instead of being fully specified
in advance.[^mitchell][^russell-norvig]

A **sample** is one observation. A **feature** is an input
variable or coordinate describing a sample. A **label** or **target** is
the answer to be predicted. This
statistical use of *feature* is different from a phonological feature:
the latter describes a property such as voicing or nasality in a sound
system; an ML feature is a data value supplied to an algorithm and may
encode a phonological property, a raw transcription, or something else.

A **model** maps input features to a prediction or
representation. Its **parameters** are values estimated from
training data. **Hyperparameters** specify how learning is carried out
and are chosen outside the parameter-fitting step; these include
settings such as the strength of a **regularization** penalty. A fitted model
does not normally change its learned parameters when it predicts a new
sample.

During **training** or **fitting**, an algorithm receives training
samples and, in supervised learning, their labels. It uses features to
estimate model parameters that reduce a chosen loss or maximize a
chosen objective. At **prediction** or **inference**, the fitted model
applies its learned mapping to new input features to produce a prediction.

Data used to compare models or choose hyperparameters form a
validation set. **Validation** may guide model selection,
feature choices, thresholds, or hyperparameter tuning. A separate
test set is held back until these choices are complete;
**testing** or **evaluation** estimates how the selected procedure
performs on held-out data. Repeatedly adjusting a
model after seeing test results turns the test set into validation data
and makes the reported evaluation optimistic. **Cross-validation**
repeats fitting and validation across several partitions
when data are limited.[^hastie]

For example, consider classifying dialect spots into dialect groups.
Each sample is one dialect spot, its pronunciation values across a
shared character set are features, and its known dialect group is the
label. Training estimates model parameters from labeled spots.
Validation can select hyperparameters, such as regularization strength.
After those choices are fixed, testing on held-out spots estimates
performance on unseen dialect spots; predictions for new spots then
assign them to a dialect group. The split must match this goal: records
from a held-out spot must not leak into training through other
characters or derived features. Successful prediction reflects
patterns in the supplied data and labels, not proof that the groups are
historically valid or that the selected features cause the grouping.

In **supervised learning**, examples have labels or targets.
**Unsupervised learning** looks for structure in data without supplied
target labels. **Semi-supervised learning** combines a smaller labeled
set with a larger unlabeled set. **Self-supervised learning** derives
training targets from the input itself.[^chapelle]

Supervised tasks include **classification**, which predicts categories,
and **regression**, which predicts numeric values. Classification may
be **binary classification** (two classes), **multiclass
classification** (one of more than two classes), or **multilabel
classification** (one sample can receive several labels
simultaneously). **Multi-output / multi-target prediction** predicts
multiple target values for each sample; the outputs may be multiple
classes, labels, or numeric values. These describe different aspects
of a task: multilabel specifies how labels can combine, while
multi-output specifies that the target has multiple components.

**Maximum likelihood estimation** (MLE) chooses
parameters that make the observed training data most probable under
the model. **Maximum a posteriori estimation** (MAP)
chooses the parameter value with the greatest posterior probability
after combining the data likelihood with a prior. A prior can favor
plausible or simpler parameter values; in common models this is related
to regularization. **Bayesian inference** applies Bayes'
rule to obtain a posterior distribution over parameters or hypotheses,
rather than only a single best value. The posterior combines prior
beliefs with the likelihood of observed data and represents remaining
uncertainty. MLE and MAP return point estimates; Bayesian analysis can
use the full posterior to quantify uncertainty in predictions and
parameters.[^bishop]

**Clustering** is one of the unsupervised tasks that groups
examples into clusters according to a chosen notion of
distance or structure. A **cluster** is a collection of samples assigned
to the same group; a **centroid** is a representative location for a cluster, often the mean
of its member vectors.[^hastie]

**Feature engineering** is the process of turning source
data into useful model inputs. A typical workflow defines the sample
unit and prediction target, inspects and cleans source values, chooses
features that are available at prediction time, encodes categories or
sequences numerically, handles missing data, and scales or selects
features when appropriate. These transformations must be fitted using
training data only, then applied unchanged to validation, test, and
future samples. The feature design is part of the method and can affect
the result as much as the classifier.[^kuhn-johnson]

Common classification metrics include **accuracy**, the
fraction of predictions that are correct; **precision**, the
fraction of predicted members of a class that truly belong to it; and
**recall**, the fraction of a class's true members recovered.
The **F1 score** is the harmonic mean of precision and
recall. A **confusion matrix** counts actual versus predicted
classes and helps reveal which classes are confused. When class
frequencies differ, **balanced accuracy** averages
per-class recall, and macro-averaged precision, recall, or F1 weights
classes equally. For a continuous prediction, common metrics include
**mean absolute error (MAE)** and **root mean squared error (RMSE)**.
When reporting
precision, recall, or F1 across multiple
classes, specify whether scores are averaged equally by class (macro),
by sample count (weighted), or over all decisions together (micro).
Metrics answer different questions; for example, overall accuracy can
obscure poor performance on a small dialect group.[^sokolova]

**Underfitting** occurs when a
model is too limited to capture patterns needed for the task.
**Overfitting** occurs when a model fits peculiarities of training data
that do not carry over to new samples.
This depends in part on the model class's **expressive capacity**: a model
with too little capacity may underfit, while one with very high capacity
may fit noise and overfit. The **VC dimension** describes the capacity
of a class of binary classifiers by measuring how many points it can
shatter (realize every possible binary labeling of those points); it
is one theoretical way to reason about the balance between
model complexity, available data, and generalization.[^vapnik]
**Generalization** is performance on data outside the fitting process.
**Data leakage** occurs when information unavailable at the real
prediction time, or information from evaluation data, influences
training or model selection. The evaluation unit and metric should be
chosen to match the intended use.

| Term | Definition |
| --- | --- |
| Machine learning | Estimating patterns or decision rules from data. |
| Expert system | A system applying explicitly encoded human rules and facts. |
| Classification | Predicting a discrete class or category for an example. |
| Regression | Predicting a numeric value. |
| Binary classification | Classification between two classes. |
| Multiclass classification | Choosing one class from more than two alternatives. |
| Multilabel classification | Assigning zero or more labels to an example. |
| Multi-output / multi-target prediction | Predicting multiple target values for each example. |
| Sample / example | One observation used for fitting or evaluation. |
| Feature | An input variable or coordinate describing a sample. |
| Label / target | The desired output used in supervised learning. |
| Model | A fitted mapping from inputs to predictions or representations. |
| Parameter | A value estimated by fitting a model to training data. |
| Hyperparameter | A setting that controls a learning procedure and is selected outside its parameter-fitting step. |
| Loss / objective | A quantity minimized or maximized during parameter fitting. |
| Training / fitting | Estimating model parameters from training data. |
| Prediction / inference | Applying a fitted model to input; in Bayesian statistics, estimating unknown quantities or their distributions. |
| Validation | Comparing or tuning a model using data separate from its fitting data. |
| Validation set | Data used for model selection or hyperparameter tuning. |
| Testing / evaluation | Measuring a selected model on held-out data. |
| Test set | Held-out data reserved for final evaluation. |
| Cross-validation | Repeated fitting and validation on different data partitions. |
| Supervised learning | Learning from examples paired with targets. |
| Unsupervised learning | Finding structure without supplied target labels. |
| Semi-supervised learning | Learning from both labeled and unlabeled examples. |
| Self-supervised learning | Learning targets derived from the input data. |
| Nonparametric method | A method whose effective complexity is not fixed by a finite-dimensional parameterization. |
| Maximum likelihood estimation (MLE) | Choosing parameters that maximize the likelihood of observed data. |
| Maximum a posteriori estimation (MAP) | Choosing the parameter value with the highest posterior probability. |
| Bayesian inference | Updating a probability distribution over hypotheses or parameters using observed data. |
| Prior | A probability distribution over parameters or hypotheses before considering the observed data. |
| Likelihood | The probability of observed data under a specified parameter value or hypothesis. |
| Posterior | The probability distribution after combining prior information with the observed data. |
| Regularization | A constraint or penalty that discourages overly complex model fits. |
| Clustering | Grouping examples by similarity without using supplied class labels. |
| Cluster | A collection of samples assigned to the same unsupervised group. |
| Centroid | A representative location for a cluster, often the mean of its member vectors. |
| Feature engineering | Designing and transforming input variables for a learning procedure. |
| Accuracy | The fraction of predictions that are correct. |
| Precision | The fraction of predicted class members that truly belong to that class. |
| Recall | The fraction of true class members correctly identified. |
| F1 score | The harmonic mean of precision and recall. |
| Confusion matrix | A table of actual and predicted class counts. |
| Balanced accuracy | The average recall across classes. |
| Macro / weighted / micro average | Averaging metric values equally across classes / by class frequency / over all decisions. |
| Mean absolute error (MAE) | The average absolute difference between predictions and observed values. |
| Root mean squared error (RMSE) | The square root of the average squared prediction error. |
| Overfitting | Fitting training-specific patterns that fail to generalize. |
| Underfitting | Failing to capture patterns needed for the task. |
| Expressive capacity | The range or complexity of patterns a model class can represent. |
| VC dimension | A measure of the capacity of a binary classifier class based on the largest set it can shatter. |
| Generalization | Performance on data not used to fit the model. |
| Data leakage | Use of evaluation or unavailable information during fitting or model selection. |
| Missing value | An unavailable observation, not an observed negative fact. |

## Representation learning

In feature engineering, **encoding** converts source values into a
numerical form a learning algorithm can use.[^kuhn-johnson] To
**encode** a categorical value, for example, one can use one-hot
encoding, which represents the category with a vector whose coordinates
indicate category membership. A **vector** is an ordered sequence of
coordinates representing a sample or entity. An **embedding** is a vector
representation learned from data; it is often dense and lower
dimensional than a one-hot vector. Its geometry reflects the training
objective and data, and should not automatically be interpreted as
linguistic or historical structure.

**Representation learning** learns useful representations from data,
rather than relying only on manually designed features. An **encoder**
maps an input to a **latent representation**. A
**decoder** maps that representation back to an output, such as a
reconstruction of the input. An **autoencoder** combines an encoder and
decoder and is trained to reproduce its input. **Reconstruction** in
this setting is the process of generating an approximation of the input
from the learned representation. The reconstructed output need not be
exact, and its quality is measured against a chosen objective. Encoders
can also be trained for other tasks, so not every encoder has a decoder
or performs reconstruction of its input.[^bengio][^goodfellow]

| Term | Definition |
| --- | --- |
| Encoding | Converting source values into a representation usable by a learning algorithm. |
| Encode | To convert source data into a representation. |
| Vector | An ordered sequence of coordinates representing a sample or entity. |
| Embedding | A dense vector representation, often learned from data. |
| Representation learning | Learning useful representations from data. |
| Encoder | A mapping from input to a representation. |
| Decoder | A mapping from a representation to an output. |
| Autoencoder | An encoder-decoder model trained to reproduce its input. |
| Reconstruction | The process of generating an approximation of an input from a representation. |

## SinComp concepts

SinComp reuses linguistic terms with data-oriented meanings. These
definitions describe what the library stores or calculates, not a
complete linguistic interpretation.

In SinComp, a **dialect** (方言) is a data unit representing a survey
location, a source record, or another internally assigned unit,
identified by a dialect ID.

A **character** (字) is an aligned data unit identified by a
character ID. It is not necessarily a written
character: it usually corresponds to one monosyllabic morpheme, but
that correspondence depends on the data source. For example, some
polyphones with semantic distinctions that can be analyzed as separate
morphemes may still share one character ID.
In some contexts, the term *character* also refers to the glyph.

A **sample** (样本) in SinComp usually represents one dialect's record
for one character. A dialect-character pair may nevertheless have
multiple samples because of multiple readings, sources, or other
source-specific distinctions; the pair is not necessarily a unique
sample identifier.

A **phone** (音) is a source-compatible pronunciation representation
associated with a dialect and character. It may represent speech at
any granularity, from an initial or final to tone or a whole syllable;
it is not necessarily a canonical IPA phone or phoneme. A **segment**
(音段) is a unit produced by dividing a phone or pronunciation
representation according to an analysis. SinComp uses the term broadly
for any such division, including units analyzed as suprasegmental
phonemes.

A **feature** (特征) is the broadest term in this vocabulary: any
property extracted from or used to describe a phone. It may be the
phone or a segment itself, or a property derived from one, such as an
initial, final, tone, medial, nucleus, coda, or other phonetic
property. This is broader than the use of feature in phonology. A
**value** (值) is a feature's value for a particular dialect and
character.

For a fixed dialect and feature, two characters are equivalent when
their phones have the same feature value. Each resulting equivalence
class of characters is a **phone category** (音类). The **phone value**
(音值) of that sound class for the feature is the shared value that
defines it. A sound class depends on the chosen dialect and feature;
its value is not necessarily a phoneme.

A **dialect vector** (方言向量) is a numerical representation of a
dialect, and a **character vector** (字向量) is a numerical
representation of a character. A **phone vector** (音向量) represents
phone or segment information as numerical coordinates; it can describe
a character's pronunciation profile under selected features. A
**dialect embedding** (方言嵌入) / **character embedding** (字嵌入) /
**phone embedding** (音嵌入) is a dense vector representing a dialect /
character / phone value respectively. Embeddings are often learned from data.

Within some dialect groups, such as Mandarin and Min, the varieties are
further divided into several subregions. Chinese dialectology has no
formal level name for this finer division; SinComp calls it a **sub group** (次方言区).

| English term | Chinese term | SinComp meaning |
| --- | --- | --- |
| Dialect | 方言 | A data unit identified by dialect ID. |
| Character | 字 | An aligned data unit, usually corresponding to one monosyllabic morpheme, identified by character ID. |
| Sample | 样本 | Usually one dialect's record for one character; multiple samples may exist for a dialect-character pair. |
| Phone | 音 | A source-compatible pronunciation representation at any granularity. |
| Segment | 音段 | A unit produced by dividing a phone or pronunciation representation. |
| Feature | 特征 | Any property used to describe or derive information from a phone. |
| Value | 值 | The value of a feature for a particular dialect and character. |
| Phone category | 音类 | An equivalence class of characters whose phones share a feature value in a dialect. |
| Phone value | 音值 | The shared feature value defining a sound class for a feature. |
| Dialect vector | 方言向量 | A numerical representation under a named estimator. |
| Character vector | 字向量 | A numerical representation of a character under a selected estimator. |
| Phone vector | 音向量 | Numerical coordinates representing a phone or a character's pronunciation profile. |
| Dialect embedding | 方言嵌入 | A dense vector representing a dialect, often learned from data. |
| Character embedding | 字嵌入 | A dense vector representing a character, often learned from data. |
| Phone embedding | 音嵌入 | A dense vector representing a phone value, often learned from data. |
| Sub group | 次方言区 | A further division within a dialect group, as used in SinComp. |

For data formats, see [Datasets](datasets.md). For estimators and vector
tables, see [Feature extraction](feature-extraction.md),
[Similarity](similarity.md), [Rules](rules.md), and
[Embeddings and decomposition](decomposition.md).

[^ipa]: International Phonetic Association, *Handbook of the International
    Phonetic Association: A Guide to the Use of the International
    Phonetic Alphabet*, Cambridge University Press, 1999.
[^catford]: J. C. Catford, *A Practical Introduction to
    Phonetics*, 2nd ed., Oxford University Press, 2001.
[^crystal]: David Crystal, *A Dictionary of Linguistics and Phonetics*,
    6th ed., Blackwell, 2008.
[^campbell]: Lyle Campbell, *Historical Linguistics: An Introduction*,
    4th ed., Edinburgh University Press, 2017.
[^crowley-bowern]: Terry Crowley and Claire Bowern, *An Introduction to
    Historical Linguistics*, 4th ed., Oxford University Press, 2010.
[^nakhleh]: Luay Nakhleh, Tandy Warnow, Don Ringe and Steven N. Evans,
    “A Comparison of Phylogenetic Reconstruction Methods on an IE
    Dataset,” *Transactions of the Philological Society* 103(2),
    2005, https://doi.org/10.1111/j.1467-968X.2005.00149.x
[^chambers-trudgill]: J. K. Chambers and Peter Trudgill, *Dialectology*,
    2nd ed., Cambridge University Press, 1998.
[^termonline]: 全国科学技术名词审定委员会，“术语在线”,
    https://www.termonline.cn
[^glottolog]: Glottolog 5.2, “Sino-Tibetan,”
    https://glottolog.org/resource/languoid/id/sino1245
[^dialect-atlas]: Raymond Hickey, ed., *The Cambridge Handbook of Areal
    Linguistics*, Cambridge University Press, 2017.
[^norman]: Jerry Norman, *Chinese*, Cambridge University Press, 1988.
[^pulleyblank]: Edwin G. Pulleyblank, *Lexicon of Reconstructed
    Pronunciation in Early Middle Chinese, Late Middle Chinese, and
    Early Mandarin*, UBC Press, 1991.
[^chao]: Yuen Ren Chao, *A Grammar of Spoken Chinese*, University of
    California Press, 1968.
[^unicode]: The Unicode Consortium, *The Unicode Standard*,
    https://www.unicode.org/versions/latest/
[^mitchell]: Tom M. Mitchell, *Machine Learning*, McGraw-Hill, 1997.
[^russell-norvig]: Stuart Russell and Peter Norvig, *Artificial
    Intelligence: A Modern Approach*, 4th ed., Pearson, 2021.
[^chapelle]: Olivier Chapelle, Bernhard Schölkopf and Alexander Zien,
    eds., *Semi-Supervised Learning*, MIT Press, 2006.
[^hastie]: Trevor Hastie, Robert Tibshirani and Jerome Friedman, *The
    Elements of Statistical Learning: Data Mining, Inference, and
    Prediction*, 2nd ed., Springer, 2009.
[^bishop]: Christopher M. Bishop, *Pattern Recognition and Machine
    Learning*, Springer, 2006.
[^kuhn-johnson]: Max Kuhn and Kjell Johnson, *Feature Engineering and
    Selection: A Practical Approach for Predictive Models*, CRC Press,
    2019.
[^sokolova]: Marina Sokolova and Guy Lapalme, “A systematic analysis of
    performance measures for classification tasks,” *Information
    Processing & Management* 45(4), 2009, pp. 427–437,
    https://doi.org/10.1016/j.ipm.2009.03.002.
[^vapnik]: Vladimir N. Vapnik, *Statistical Learning Theory*, Wiley,
    1998.
[^bengio]: Yoshua Bengio, Aaron Courville and Pascal Vincent,
    “Representation Learning: A Review and New Perspectives,” *IEEE
    Transactions on Pattern Analysis and Machine Intelligence* 35(8),
    2013, pp. 1798–1828, https://doi.org/10.1109/TPAMI.2013.50.
[^goodfellow]: Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep
    Learning*, MIT Press, 2016.
