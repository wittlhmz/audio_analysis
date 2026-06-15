# Projektplan: scikit-learn Algorithmen auf der Musikdatenbank

Datenbasis: 834 Songs · 12 Genres · Features: Genre, Jahr, Länge

---

## Status-Legende
- ✅ Fertig
- 🔲 Offen

---

## 1. Clustering — kein Label nötig

> Pfad im Cheat Sheet: kein labeled data → number of categories known/unknown → <10K samples

| Status | Algorithmus | Ziel |
|--------|-------------|------|
| ✅ | **KMeans** (k=7) | Songs nach Genre/Jahr/Länge gruppieren |
| ✅ | **MeanShift** | Cluster ohne vorgegebenes k finden — wie viele "natürliche" Gruppen gibt es? |
| ✅ | **GMM** (Gaussian Mixture Model) | Wie KMeans, aber mit Wahrscheinlichkeiten statt harten Grenzen |
| ✅ | **Spectral Clustering** | Nicht-lineare Cluster-Strukturen erkennen |

**Konkreter Output:** Vergleichsplot aller vier Algorithmen nebeneinander (je 2D via PCA).

---

## 2. Dimensionsreduktion — Struktur sichtbar machen

> Pfad im Cheat Sheet: just looking → <10K samples → Isomap, Spectral Embedding, LLE, Randomized PCA

| Status | Algorithmus | Ziel |
|--------|-------------|------|
| ✅ | **PCA** (Standard, 2D) | Varianz-maximierende Projektion |
| ✅ | **Randomized PCA** | Schnellere PCA-Variante für größere Feature-Räume |
| ✅ | **t-SNE** | Ähnliche Songs nah beieinander darstellen (sehr populär in Portfolios) |
| ✅ | **Isomap** | Geodätische Abstände statt euklidische — curved structure |
| ✅ | **Spectral Embedding** | Graph-basierte Projektion der Song-Ähnlichkeiten |
| ✅ | **LLE** (Locally Linear Embedding) | Lokale Nachbarschaften linearisieren |

**Konkreter Output:** 6 Subplots mit denselben Daten, alle eingefärbt nach Genre — direkt vergleichbar.

---

## 3. Klassifikation — Genre vorhersagen

> Pfad im Cheat Sheet: labeled data → predicting a category → <100K samples

Genre ist das Label, Jahr + Länge sind Features. Klein, aber zeigt den vollen ML-Workflow.

| Status | Algorithmus | Anmerkung |
|--------|-------------|-----------|
| ✅ | **KNeighbors Classifier** | Einfacher Einstieg, sehr interpretierbar |
| ✅ | **Linear SVC** | Schnell, gut bei wenigen Features |
| ✅ | **SVC** (kernel='rbf') | Nicht-linearer Kernel für komplexere Trennungen |
| ✅ | **Ensemble Classifiers** (RandomForest) | Robust, gibt Feature Importances aus |

**Konkreter Output:** Confusion Matrix + Classification Report (Precision, Recall, F1 pro Genre).
Hinweis: Klassen sind unbalanciert (Electronic: 339 vs R&B: 1) → Stratified Split + class_weight nötig.

---

## 4. Regression — Songlänge oder Jahr vorhersagen

> Pfad im Cheat Sheet: labeled data → predicting a quantity → <100K samples

Ziel: Lässt sich die Länge eines Songs aus Genre + Jahr vorhersagen?

| Status | Algorithmus | Anmerkung |
|--------|-------------|-----------|
| ✅ | **Ridge Regression** | Regularisierter Baseline-Ansatz |
| ✅ | **Lasso / ElasticNet** | Feature Selection durch L1-Regularisierung |
| ✅ | **SVR** (kernel='linear') | Support Vector Regression |

**Konkreter Output:** Predicted vs. Actual Plot + R²-Score. Erwartung: niedriger R² — zeigt ehrlich die Grenzen der Daten.

---

## 5. Datenaufbereitung (Voraussetzung für 3 & 4)

| Status | Aufgabe |
|--------|---------|
| ✅ | `year`-Spalte von TEXT zu INT konvertieren (aktuell als String gespeichert) |
| ✅ | Kleine Klassen (Reggae: 1, R&B: 1) für Klassifikation droppen |
| ✅ | Train/Test-Split mit `stratify=genre` |

---

## Empfohlene Reihenfolge

1. **Datenaufbereitung** (5) — brauchen alle nachfolgenden Schritte
2. **Dimensionsreduktion** (2) — visuell eindrucksvoll, sofort sichtbares Ergebnis
3. **Clustering-Vergleich** (1) — baut auf vorhandener Arbeit auf
4. **Klassifikation** (3) — kompletter ML-Workflow, recruitertechnisch wichtig
5. **Regression** (4) — rundet das Bild ab, zeigt kritisches Denken (schlechter R²)
