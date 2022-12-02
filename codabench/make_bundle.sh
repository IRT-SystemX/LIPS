#!/bin/bash

BUNDLE_NAME="lips_bundle"

ORIG_FOLDER=$BUNDLE_NAME
DEST_FOLDER="$BUNDLE_NAME"_codabench
DEST_ZIP="$BUNDLE_NAME"_codabench.zip

echo -e "Destination folder is: $DEST_FOLDER"
echo -e "Output zip is: $DEST_ZIP\n\n"

# cleaning of the repo before the starting kit
rm -rf $DEST_ZIP
find $ORIG_FOLDER -type d -name '__pycache__' | xargs -i rm -rf {}

# Make the lips_bundle zip
echo -e "Creating $DEST_ZIP ..."
rm -rf $DEST_FOLDER; mkdir $DEST_FOLDER
cp $ORIG_FOLDER/*.{md,yaml,png} $DEST_FOLDER
# substitute symbolic links
cp -LR benchmark_config $DEST_FOLDER
cp -R ingestion_* $DEST_FOLDER
cp -R scoring_* $DEST_FOLDER
#cp -R input_data* $DEST_FOLDER
#cp -R reference_data* $DEST_FOLDER
(cd $DEST_FOLDER && zip -rq ../$DEST_ZIP .)
rm -rf $DEST_FOLDER
echo -e "Done $DEST_ZIP\n"
echo -e "$DEST_ZIP ready for codabench"
