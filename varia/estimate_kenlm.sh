# kenLM repo: https://github.com/kpu/kenlm
DATADIR="/mnt/storage/scratch2/siclemat/nnlm2019data"
OUTDIR="/mnt/storage/clfiles/users/makarov/kenlm_lms"
ORDER=3
DATE="20190506"
KENPATH="/mnt/storage/clfiles/users/makarov/kenlm/build/bin"
for LANG in is  de  es  hu  pt  se  sl  sv; do
echo "ESTIMATING N-GRAM LM for: $LANG"
TEXT="$DATADIR"/"$LANG"/normalized/wiki.tokens.txt
ARPA="$OUTDIR"/"$LANG"_o"$ORDER"_"$DATE".arpa
KENLM="$OUTDIR"/"$LANG"_o"$ORDER"_"$DATE".klm
"$KENPATH"/lmplz -o $ORDER -S 13% -T /tmp < $TEXT > $ARPA
"$KENPATH"/build_binary $ARPA $KENLM
done
