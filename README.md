# notable-ML
Concerns the development of the machine learning backend inference for the Notable application

Notice of file structure changes:
```md
.ML
+-- <segmenter>
|   +-- [output](location of sliced sheets) 
|   +-- slicer.py
+-- <Data>
|   +-- [raw](default location of .wav files)
|   +-- semantic_vocabulary.txt
+-- [CMD.ipynb](Start here, launch predict-loop instead of ctc predict)
+-- ctcpredict-loop 
+-- [requirements.txt](install packages listed)
+-- totalrequirements.txt
----------------------
<Default-arguments
-sheet "path to the sheet"
-image "path to the output of segmenter"
-model "path to the model"
-vocabulary "path to the semantic vocabulary"
-type "type of output" = clean/raw/perfect
-seq "single wav output or combined" = true/false
======================
```

### Kanban:
  https://github.com/users/Aroueterra/projects/2
