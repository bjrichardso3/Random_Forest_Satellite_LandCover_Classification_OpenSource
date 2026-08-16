This document details the Powershell commands required to run the Random Forest Model Training and Classification Pipeline Scripts from the command line



\***Bold represents user entered parameters**





**\*Ensure environment is activated before running the script: micromamba activate sat\_pipeline**



The training workflow performs the following automatically:



1. Downloads the satellite imagery for the training year.

2\. Preprocesses the satellite scenes to the common 30 m grid.

3\. Creates the annual feature composite.

4\. Creates target grids at each requested resolution.

5\. Resamples the authoritative CEH reference data to each target grid.

6\. Creates spatially and categorically stratified reference samples.

7\. Creates the 70/30 training/testing split.

8\. Performs spatial cross-validation using spatial blocks.

9\. Trains the final Random Forest model using the training samples.

10\. Saves the model, metadata and cross-validation results.

11\. Evaluates the held-back 30% test samples automatically.







National Scale Model Training:

python -m random\_forest\_model.run\_training `

&#x20;   --aoi "**Add file path of national area of interest shapefile here**" `

&#x20;   --reference "**Add file path of authoritative land cover dataset here (i.e. CEH Land Cover)**" `

&#x20;   --year **2022** `

&#x20;   --resolutions 50 250 1000 `

&#x20;   --samples 20000 `

&#x20;   --min-per-class 100



Catchment Scale Model Training:

python -m random\_forest\_model.run\_training `

&#x20;   --aoi "**Add file path of catchment area of interest shapefile here**" `

&#x20;   --reference "**Add file path of authoritative land cover dataset here (i.e. CEH Land Cover)**" `

&#x20;   --year **2022** `

&#x20;   --resolutions 50 250 1000







The internal 70/30 evaluation is automatically performed by run\_training.py.



The reference samples are divided into:



70% training samples

30% held-back test samples



The Random Forest is trained using only the training samples, and the held-back test samples are subsequently evaluated.



The 70 / 30 internal evaluation is performed automatically for each resolution requested during training.



The training workflow also performs spatial cross-validation using StratifiedGroupKFold, with the spatial block\_id used as the grouping variable.



The resulting model package contains:



training sample count

training class distribution

spatial cross-validation results

mean cross-validation metrics

out-of-bag score

model parameters

feature names

training year

AOI name

resolution



The separate evaluation output contains CEH-level and SHETRAN-level results, including:



accuracy

balanced accuracy

macro F1

weighted F1

Cohen's kappa

confusion matrix

normalised confusion matrix

class-level precision

class-level recall

class-level F1

class-level IoU







Model Prediction External Evaluation Run (external, i.e. different catchment or year)



External evaluation is used to assess an already-trained Random Forest against reference data from a different year or a different spatial extent.



The external evaluation workflow:



1. Loads the trained Random Forest model.

2\. Reads the model's training year and AOI metadata.

3\. Downloads satellite imagery for the external year.

4\. Preprocesses the imagery.

5\. Creates the annual feature composite.

6\. Resamples the satellite features to the requested target resolution.

7\. Resamples the external CEH reference raster to the same target grid.

8\. Generates external reference samples.

9\. Labels those samples as external\_test.

10\. Evaluates the trained model against those samples.

11\. Produces CEH and SHETRAN evaluation results.



python -m random\_forest\_model.run\_external\_evaluation `

&#x20;   --aoi "**Add file path to area of interest shapefile here**" `

&#x20;   --reference "**Add file path to authoritative land cover dataset (i.e. CEH Land cover) (for external evaluation year which could be different than what was trained on)**" `

&#x20;   --year **2023** `

&#x20;   --resolution **50** `

&#x20;   --model "**Add file path to random forest model used .joblib**"







Pipeline Classification Run:



Once a Random Forest model has been trained, it can be used to classify annual satellite feature composites.



The classification workflow:



1. Downloads satellite imagery for the requested period.

2\. Preprocesses the satellite scenes.

3\. Creates annual feature composites.

4\. Resamples the feature composite to the requested target resolution.

5\. Applies the trained Random Forest model.

6\. Produces CEH classification.

7\. Converts CEH predictions to SHETRAN classes.

8\. Produces a Random Forest prediction-confidence raster.



python -m pipeline.main\_run\_pipeline `

&#x20;   --aoi "**Add file path of area of interest shapefile here**" `

&#x20;   --start-year **2023** `

&#x20;   --end-year **2023** `

&#x20;   --resolution **50** `

&#x20;   --model **"Add file path to desired random forest model .joblib"**

