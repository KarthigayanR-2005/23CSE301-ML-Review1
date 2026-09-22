"""algorithm_content.py - the per-algorithm write-ups used by the notebooks.

One source of truth for: what each algorithm is (plain words), how it works
(numbered steps), which settings matter, the viva note, and the result we
measured on our data.

`scripts/build_notebooks.py` imports these and turns each entry into a notebook
section: the markdown explanation, then the code that trains THAT model on our
data, then its output.

Scope: REVIEW 1 - the 10 regression algorithms and the 5 classification Part A
algorithms. Part B and clustering belong to Review 2.
"""
from __future__ import annotations

REGRESSION = [
    dict(n=1, slug="linear_regression", title="1. Linear Regression",
         imports="from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import Pipeline",
         estimator="LinearRegression()", scale="True", poly="",
         what="Draws the single best straight line (in 5 dimensions, a flat plane) through the data. It assumes the target is just a weighted sum of the inputs plus a constant.",
         how=["Give every input feature a weight, starting from anything.",
              "For each student, predict: weight1 x hours + weight2 x previous score + ... + intercept.",
              "Measure how wrong the predictions are, using the sum of squared errors.",
              "Solve, in one step of algebra (the 'normal equation'), for the weights that make that total error as small as it can be.",
              "Those weights are the trained model. There is nothing to iterate - the answer is exact."],
         settings="None worth tuning. That is the point of a baseline: it has no knobs to hide behind.",
         viva="A coefficient says: if this feature goes up by one standard deviation and everything else stays fixed, the predicted Performance Index changes by this much. Our inputs are standardised, so the coefficients are directly comparable to each other.",
         result="Test R2 = 0.988983, RMSE = 2.02 points. Ranked 2nd of 10 - a plain straight line beat eight more complicated models.",
         extra='''
# --- ALGORITHM-SPECIFIC: read the coefficients (PDF note: "interpret coefficients")
from src.regression_models import coefficient_table
coefs = coefficient_table(model)
print("\\n  Coefficients (change in Performance Index per 1 standard deviation):")
print(coefs.to_string(index=False))
print(f"  intercept: {coefs.attrs['intercept']:.4f}")'''),

    dict(n=2, slug="ridge_regression", title="2. Ridge Regression",
         imports="from sklearn.linear_model import Ridge\nfrom sklearn.pipeline import Pipeline",
         estimator="Ridge(alpha=1.0, random_state=42)", scale="True", poly="",
         what="Linear Regression with a leash. It still fits a straight line, but it is penalised for using large weights, which stops any single feature from dominating.",
         how=["Set up the same squared-error total as Linear Regression.",
              "Add a penalty term: alpha x (sum of all the weights squared).",
              "Minimise error + penalty together, not error alone.",
              "Because big weights now cost something, the solution shrinks every weight toward zero - but never exactly to zero.",
              "alpha controls the leash length: alpha = 0 is plain Linear Regression, huge alpha forces all weights near zero."],
         settings="alpha - the strength of the penalty. We searched 0.01, 0.1, 1.0, 10.0, 100.0 and the best was 0.1, i.e. almost no penalty needed.",
         viva="This is L2 regularisation. It helps when features are correlated or when you have more features than data. Here neither is true, which is exactly why it scored the same as plain Linear Regression.",
         result="Test R2 = 0.988982. Tuning alpha moved the score in the 5th decimal place.",
         extra=""),

    dict(n=3, slug="lasso_regression", title="3. Lasso Regression",
         imports="from sklearn.linear_model import Lasso\nfrom sklearn.pipeline import Pipeline",
         estimator="Lasso(alpha=0.01, random_state=42, max_iter=10000)", scale="True", poly="",
         what="Like Ridge, but its penalty can switch features off completely by setting their weight to exactly zero. It does feature selection while it fits.",
         how=["Same squared-error total as before.",
              "Add a penalty of alpha x (sum of the ABSOLUTE values of the weights).",
              "Absolute value has a sharp corner at zero, and that corner makes it possible for a weight to land exactly on zero rather than merely near it.",
              "Solve iteratively (coordinate descent) - unlike Ridge there is no one-step formula.",
              "Any feature whose weight ends at zero has been dropped from the model."],
         settings="alpha - larger alpha zeroes out more features. We searched 0.001, 0.01, 0.1, 1.0; best was 0.001.",
         viva="L1 versus L2 is the classic question. L1 (Lasso) gives sparsity - it deletes features. L2 (Ridge) only shrinks them. On our data Lasso zeroed NOTHING: 0 of 5 coefficients. That means every one of our five inputs is carrying real signal.",
         result="Test R2 = 0.988969. Zero features dropped - reported in results/tables/regression_lasso_sparsity.json.",
         extra='''
# --- ALGORITHM-SPECIFIC: feature sparsity (PDF note: "observe feature sparsity")
from src.regression_models import sparsity_report
print("\\n  Sparsity:", sparsity_report(model))'''),

    dict(n=4, slug="elasticnet_regression", title="4. ElasticNet Regression",
         imports="from sklearn.linear_model import ElasticNet\nfrom sklearn.pipeline import Pipeline",
         estimator="ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)",
         scale="True", poly="",
         what="Ridge and Lasso mixed together. You choose how much of each penalty to apply.",
         how=["Same squared-error total.",
              "Add BOTH penalties: l1_ratio x (Lasso penalty) + (1 - l1_ratio) x (Ridge penalty), all scaled by alpha.",
              "l1_ratio = 1 makes it pure Lasso; l1_ratio = 0 makes it pure Ridge; 0.5 is an even blend.",
              "Solve iteratively, as with Lasso.",
              "You get some feature dropping from the L1 part and some smooth shrinkage from the L2 part."],
         settings="alpha (overall penalty strength) and l1_ratio (the mix). We searched alpha in 0.001-1.0 and l1_ratio in 0.1/0.5/0.9.",
         viva="Why exist at all, if Ridge and Lasso already do? Because when features come in correlated groups, pure Lasso arbitrarily keeps one and drops the rest, whereas ElasticNet tends to keep or drop the group together.",
         result="Test R2 = 0.988886 - the weakest of the four linear models here, by a hair.",
         extra=""),

    dict(n=5, slug="polynomial_regression", title="5. Polynomial Features + Linear Regression",
         imports="from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import PolynomialFeatures",
         estimator="LinearRegression()", scale="True",
         poly='    ("poly", PolynomialFeatures(degree=2, include_bias=False)),\n',
         what="Linear Regression given extra, invented columns: every feature squared, and every pair of features multiplied together. The model is still linear in its weights, but it can now bend.",
         how=["Start with the 5 preprocessed input columns.",
              "PolynomialFeatures(degree=2) manufactures new columns: hours^2, sleep^2, hours x sleep, and so on. 5 columns become 20.",
              "Hand all 20 columns to an ordinary Linear Regression.",
              "It fits weights for the new columns exactly as it did for the originals.",
              "The result is a curved surface in the original feature space, even though the fitting maths never changed."],
         settings="degree - how far to expand. Higher degree means far more columns and a strong risk of overfitting.",
         viva="Degree 1 gave 5 terms and test R2 0.98898. Degree 2 gave 20 terms and 0.98899. Degree 3 gave 55 terms and 0.98896 - WORSE despite 11x the columns. That is overfitting made visible: more flexibility, worse performance on unseen data.",
         result="Test R2 = 0.988989. Ranked 1st of 10 - but it beat plain Linear Regression only in the 5th decimal, using 4x the terms.",
         extra='''
# --- ALGORITHM-SPECIFIC: compare degrees (PDF note: "compare degrees")
from src.regression_models import polynomial_degree_comparison
print("\\n  Degree comparison:")
print(polynomial_degree_comparison(NUM, CAT, X_train, y_train, X_test, y_test,
                                   degrees=(1, 2, 3)).to_string(index=False))'''),

    dict(n=6, slug="decision_tree_regressor", title="6. Decision Tree Regressor",
         imports="from sklearn.tree import DecisionTreeRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="DecisionTreeRegressor(max_depth=8, random_state=42)", scale="False", poly="",
         what="A flowchart of yes/no questions. Follow the answers down to a leaf, and the prediction is the average target value of the training students who ended up in that leaf.",
         how=["Start with all 8,000 training rows in one group.",
              "Try every feature and every possible cut point (e.g. 'Previous Scores < 62?').",
              "Pick the single cut that most reduces the variance of the target inside the two resulting groups.",
              "Split, then repeat the whole procedure independently on each group.",
              "Stop at max_depth, or when a group gets too small. Each final group is a leaf, and its prediction is the mean target of its members."],
         settings="max_depth - how many questions deep the tree may go. Deeper means it can memorise the training data. We searched 3, 5, 8, 12, None plus min_samples_leaf.",
         viva="Note scale=False. Trees do not care about feature scale at all, because a cut at 'hours < 5' means the same thing whatever units hours are in. Leaving them unscaled also keeps the tree readable in original units.",
         result="Test R2 = 0.983901, ranked 9th. A single tree predicts in steps, but the true relationship here is smooth - so steps cost accuracy.",
         extra='''
# --- ALGORITHM-SPECIFIC: feature importance (PDF note: "show feature importance")
from src.regression_models import tree_importances
print("\\n  Feature importance:")
print(tree_importances(model).round(4).to_string())'''),

    dict(n=7, slug="random_forest_regressor", title="7. Random Forest Regressor",
         imports="from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)",
         scale="False", poly="",
         what="300 different decision trees, each deliberately trained on a slightly different view of the data, and their predictions averaged.",
         how=["Draw a random sample of the training rows WITH replacement (bootstrap) - this tree sees a different dataset.",
              "Grow a decision tree on it, but at every split consider only a random subset of the features, not all of them.",
              "That double randomness makes each tree wrong in a different direction.",
              "Repeat 300 times, independently. The trees never see each other.",
              "To predict, ask all 300 trees and average their answers. Individual errors cancel out; shared signal survives."],
         settings="n_estimators (how many trees - more is steadier but slower) and max_depth. We searched 100/300/500 trees.",
         viva="Look at the overfitting gap: train R2 = 0.9975 but test R2 = 0.9862. The forest has partly memorised the training set. Averaging 300 trees controls that, but does not remove it.",
         result="Test R2 = 0.986175, ranked 7th. Slowest model in the track at 3.4 seconds, and beaten by a one-line linear fit.",
         extra='''
# --- ALGORITHM-SPECIFIC: feature importance
from src.regression_models import tree_importances
print("\\n  Feature importance (averaged over 300 trees):")
print(tree_importances(model).round(4).to_string())'''),

    dict(n=8, slug="gradient_boosting_regressor", title="8. Gradient Boosting Regressor",
         imports="from sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, random_state=42)",
         scale="False", poly="",
         what="Trees built one after another, where each new tree's job is to fix the mistakes left by all the trees before it.",
         how=["Start with a single flat guess: the mean Performance Index of the training set.",
              "Compute the residual for every row - how far off that guess was.",
              "Train a SMALL tree to predict those residuals, i.e. to predict the error itself.",
              "Add a fraction of that tree's output (the learning_rate) to the running prediction.",
              "Recompute the new, smaller residuals and repeat 200 times. Each tree corrects what remains."],
         settings="learning_rate (how much of each correction to accept - small is safer, needs more trees) and n_estimators. We searched 0.03/0.1/0.2 and 100/200/400.",
         viva="The difference from Random Forest is the key question. Random Forest builds trees in PARALLEL and averages; Gradient Boosting builds them in SEQUENCE, each fixing the last. Boosting usually wins, and here it did - 0.9884 versus 0.9862.",
         result="Test R2 = 0.988411, ranked 6th, and the best non-linear model.",
         extra=""),

    dict(n=9, slug="support_vector_regressor", title="9. Support Vector Regressor (SVR)",
         imports="from sklearn.svm import SVR\nfrom sklearn.pipeline import Pipeline",
         estimator="SVR(C=1.0, kernel='rbf')", scale="True", poly="",
         what="Fits a tube around the data and tries to get as many points as possible inside it. Errors smaller than the tube's width are treated as no error at all.",
         how=["Choose a tube width, epsilon. Any prediction within epsilon of the truth counts as correct.",
              "Find the flattest possible function that keeps most points inside the tube.",
              "Points that fall outside are penalised, with C controlling how harshly.",
              "The 'kernel trick' lets it fit a curved function without ever computing the curved coordinates - it only needs distances between points.",
              "Only the points on or outside the tube boundary (the 'support vectors') shape the final model. The rest are ignored entirely."],
         settings="C (penalty for points outside the tube - high C means fit hard, risk overfitting) and kernel ('rbf' curved, 'linear' straight). We searched C in 0.1/1/10 and both kernels.",
         viva="Why scale=True and why it matters most here: the RBF kernel is built on distances between points. An unscaled feature with a big range would swamp the distance calculation and the model would effectively ignore everything else.",
         result="Test R2 = 0.985924, ranked 8th. Took 2 seconds, versus 0.014 for Linear Regression, for a worse score.",
         extra=""),

    dict(n=10, slug="knn_regressor", title="10. K-Nearest Neighbors Regressor",
         imports="from sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="KNeighborsRegressor(n_neighbors=5)", scale="True", poly="",
         what="No training at all. To predict for a new student, find the 5 most similar students in the training data and average their Performance Index.",
         how=["'Fitting' just stores the training rows. Nothing is learned.",
              "When a prediction is requested, compute the distance from the new row to every stored training row.",
              "Sort those distances and keep the k smallest.",
              "Average the target values of those k neighbours - that is the prediction.",
              "With weights='distance', closer neighbours count for more than farther ones."],
         settings="n_neighbors (k) - small k is jumpy and noise-sensitive, large k over-smooths. We searched 3/5/9/15/25 and both weighting schemes.",
         viva="Scaling is not optional here, it is the whole ballgame. Distance is the model. If Previous Scores (range 40-99) were left unscaled against Sleep Hours (range 4-9), 'similar student' would mean 'similar previous score' and nothing else.",
         result="Test R2 = 0.976841 - LAST of 10. It is the only model that cannot extrapolate: it can never predict a value outside the range of its training neighbours.",
         extra=""),
]

CLASSIFICATION = [
    dict(n=1, slug="logistic_regression", title="A1. Logistic Regression  [Part A, Review 1]",
         imports="from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import Pipeline",
         estimator="LogisticRegression(max_iter=2000, random_state=42)", scale="True",
         what="Despite the name, this is a classifier. It computes a weighted sum of the inputs, then squashes that number into a probability between 0 and 1 with the sigmoid function.",
         how=["Compute z = weight1 x torque + weight2 x speed + ... + intercept.",
              "Squash it: probability = 1 / (1 + e^-z). Large positive z gives a probability near 1; large negative gives near 0.",
              "Find the weights that make the training labels most likely (maximum likelihood), by iterative optimisation.",
              "To predict a class, compare the probability to a threshold - 0.5 by default.",
              "The threshold is a CHOICE, not part of the model. Lowering it catches more failures at the cost of more false alarms."],
         settings="C - the inverse of regularisation strength; class_weight='balanced' makes rare failures count more. We searched C in 0.01-10 and both weightings.",
         viva="Exponentiating a coefficient gives an ODDS RATIO: how much the odds of failure multiply per one standard deviation of that feature. See results/tables/classification_logreg_odds.csv.",
         result="F1(weighted) = 0.9560, ROC-AUC = 0.8994, but Recall on failures = 0.1029. It caught 7 of 68 real failures while scoring 96.75% accuracy - the single best illustration of why accuracy is the wrong metric here.",
         extra='''
# --- ALGORITHM-SPECIFIC: odds ratios (PDF note: "interpret coefficients/odds")
from src.classification_models import odds_table
o = odds_table(model)
print("\\n  Coefficients and odds ratios (per 1 standard deviation):")
print(o.to_string(index=False))'''),

    dict(n=2, slug="knn_classifier", title="A2. K-Nearest Neighbors  [Part A, Review 1]",
         imports="from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="KNeighborsClassifier(n_neighbors=5)", scale="True",
         what="Find the 5 most similar machines in the training data and take a majority vote on whether they failed.",
         how=["Store the training rows. No model is fitted.",
              "For a new machine, measure the distance to every training machine.",
              "Keep the k closest.",
              "Count their labels. The majority label wins.",
              "The predicted probability is simply the fraction of those k neighbours that failed - which is why with k=5 the only possible probabilities are 0, 0.2, 0.4, 0.6, 0.8, 1.0."],
         settings="n_neighbors (k) and the distance metric. We compared euclidean, manhattan and chebyshev.",
         viva="Coarse probabilities are why its ROC-AUC (0.8291) is the weakest in Part A even though its accuracy looks fine. AUC needs finely graded scores to rank cases; KNN with k=5 offers only six possible values.",
         result="F1(weighted) = 0.9679, Recall(failure) = 0.2941, ROC-AUC = 0.8291.",
         extra='''
# --- ALGORITHM-SPECIFIC: distance metrics (PDF note: "discuss distance metrics")
from src.classification_models import knn_distance_metric_comparison
print("\\n  Distance metric comparison:")
print(knn_distance_metric_comparison(NUM, CAT, X_train, y_train,
                                     X_test, y_test).to_string(index=False))'''),

    dict(n=3, slug="gaussian_naive_bayes", title="A3. Gaussian Naive Bayes  [Part A, Review 1]",
         imports="from sklearn.naive_bayes import GaussianNB\nfrom sklearn.pipeline import Pipeline",
         estimator="GaussianNB()", scale="True",
         what="Uses Bayes' theorem to flip the question around. Instead of asking 'given these readings, will it fail?', it asks 'if a machine were failing, how likely are these readings?' - and combines that with how common failure is overall.",
         how=["For each class separately, compute the mean and variance of every feature from the training data.",
              "That gives a bell curve per feature per class.",
              "For a new machine, read off how likely each of its readings is under each class's bell curve.",
              "MULTIPLY those likelihoods together - this is the 'naive' step, and it assumes the features are independent given the class.",
              "Multiply by the class's base rate (3.39% for failure) and pick whichever class scores higher."],
         settings="Effectively none. GaussianNB is parameter-free; it just measures means and variances.",
         viva="The independence assumption is the exam question. Air temperature and process temperature are strongly correlated in our data - physically, the process is heated by its surroundings. Naive Bayes multiplies their evidence as if they were unrelated, so it double-counts one fact. Its F1 of 0.9506 is the lowest in Part A, and that is why.",
         result="F1(weighted) = 0.9506 - last of the five Part A models. Precision on failures 0.25: three quarters of its failure alarms were false. Do NOT substitute MultinomialNB or BernoulliNB; the PDF names Gaussian specifically.",
         extra=""),

    dict(n=4, slug="decision_tree_classifier", title="A4. Decision Tree Classifier  [Part A, Review 1]",
         imports="from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="DecisionTreeClassifier(max_depth=6, random_state=42)", scale="False",
         what="A flowchart of yes/no questions about the sensor readings, ending in a verdict of fail or no-fail.",
         how=["Begin with all 8,000 training machines in one group.",
              "Try every feature and cut point, and score each by how much it purifies the groups - measured by Gini impurity.",
              "A perfect split puts all failures on one side and all healthy machines on the other.",
              "Take the best split, then repeat on each resulting branch.",
              "Stop at max_depth. Each leaf predicts the majority class of its training members, and the probability is the class proportion in that leaf."],
         settings="max_depth, min_samples_leaf, class_weight. We searched depth 3/5/8/12/None.",
         viva="This is the only Part A model you can literally read. results/figures/classification/clf_decision_tree.png shows the top 3 levels - point at the first split and state the physical rule it found. It is also our best Part A model, so the rule it learned is worth quoting.",
         result="F1(weighted) = 0.9714, Recall(failure) = 0.3824 - the best Part A model on BOTH weighted F1 and failure recall, catching nearly 4x what Logistic Regression does.",
         extra='''
# --- ALGORITHM-SPECIFIC: draw the tree (PDF note: "visualise the tree")
from src import plotting as pl
p = pl.decision_tree_figure(model.named_steps["model"],
                            model.named_steps["prep"].get_feature_names_out(),
                            ["No Failure", "Failure"],
                            "example_decision_tree", "classification", max_depth=3)
print(f"\\n  Tree diagram saved to {p}")'''),

    dict(n=5, slug="support_vector_classifier", title="A5. Support Vector Classifier (SVC)  [Part A, Review 1]",
         imports="from sklearn.svm import SVC\nfrom sklearn.pipeline import Pipeline",
         estimator="SVC(C=1.0, kernel='rbf', probability=True, random_state=42)", scale="True",
         what="Draws the boundary between failing and healthy machines that leaves the widest possible empty margin on both sides.",
         how=["Look for a surface separating the two classes.",
              "Among all surfaces that separate them, prefer the one with the largest gap to the nearest point on each side.",
              "Allow some points to sit inside the margin or on the wrong side, penalised by C.",
              "The RBF kernel measures similarity by distance, letting the boundary curve without computing curved coordinates explicitly.",
              "Only the points nearest the boundary - the support vectors - define it. Moving a far-away point changes nothing."],
         settings="C (low C means a wide, forgiving margin; high C means fit the training data hard) and kernel. probability=True is needed for ROC-AUC and costs extra fitting time.",
         viva="Precision 0.875 with Recall 0.2059 - it is very careful. When it cries failure it is almost always right, but it stays quiet about 54 of the 68 real failures. Wide-margin fitting on an imbalanced dataset pushes the boundary toward the majority class.",
         result="F1(weighted) = 0.9635, ROC-AUC = 0.9468.",
         extra="")
]
