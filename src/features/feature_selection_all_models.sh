source .env

declare -a arr=("DT" "RF" "XGB" "MLP" "LR")
declare -a berr=("paper")

for model in "${arr[@]}"
do
for featureset in "${berr[@]}"
do

   echo "Model $model on $featureset"
   trainpath="$FEATURE_DIR/$featureset/train.csv"
   echo "$trainpath"

   if [ "$model" = "MLP" ]; then
       jobs=1
   else
       jobs=-1
   fi

   python -m src.features.create_paper_feature_set \
   -m "$model" -t 500 \
   -p "$MODEL_RESULTS" \
   -tf "$trainpath" \
   -j "$jobs" -pen 0.0005

done
done