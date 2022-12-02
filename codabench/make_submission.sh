#!/bin/bash

CODE_SUBMISSION=${1:-"code_submission"}

DEST_FOLDER="$CODE_SUBMISSION"_codabench
DEST_ZIP=lips_"$CODE_SUBMISSION"_codabench.zip

echo -e "Destination folder is: $DEST_FOLDER"
echo -e "Output zip is: $DEST_ZIP\n\n"

# Cleaning of the repo before the starting kit
rm -f $DEST_ZIP

# Make the submission bundle zip
echo -e "Creating $DEST_ZIP ..."
rm -rf $DEST_FOLDER; mkdir $DEST_FOLDER
cp -R $CODE_SUBMISSION/* $DEST_FOLDER
(cd $DEST_FOLDER && zip -rq ../$DEST_ZIP .)
rm -rf $DEST_FOLDER
echo -e "Done $DEST_ZIP\n"
echo -e "$DEST_ZIP ready for codabench"
