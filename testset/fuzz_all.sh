for f in 40 50 60 70 75 80 85 90 95 100 101
do
	python same_fuzz.py dev.aken.ak.txt dev.aken.tr.txt aken.en.txt dev.aken.en.txt $f nosub aken  > same_fuzz_in_"$f"_nosub_aken.log &

python same_fuzz.py dev.aken.ak.txt dev.aken.tr.txt aken.en.txt dev.aken.en.txt $f sub aken  > same_fuzz_in_"$f"_sub_aken.log &
python same_fuzz.py dev.aken.ak.txt dev.aken.tr.txt aken.en.txt dev.aken.en.txt $f both aken  > same_fuzz_in_"$f"_both_aken.log &

done
