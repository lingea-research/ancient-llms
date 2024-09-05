for f in 40 50 60 70 75 80 85 90 95 100 101
do
	python same_fuzz.py dev.suen.su.txt dev.suen.su.txt suen.en.txt dev.suen.en.txt $f nosub suen > same_fuzz_in_"$f"_nosub_suen.log &

python same_fuzz.py dev.suen.su.txt dev.suen.su.txt suen.en.txt dev.suen.en.txt $f sub suen > same_fuzz_in_"$f"_sub_suen.log &
python same_fuzz.py dev.suen.su.txt dev.suen.su.txt suen.en.txt dev.suen.en.txt $f both suen > same_fuzz_in_"$f"_both_suen.log &

done
