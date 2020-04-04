all:

	gcc -O3 -o fast_mmf.out -fopenmp  -mcmodel=large \
	xgft_tr.c \
	model_utils.c \
	xgft_utils.c  \
	mmf_engine.c  \
	driver.c


clean:
	rm -f *~
	rm -f *.o
	rm -f \#*\#
	rm -f *.out
	rm -f *.x
	rm -f *.log
	rm -f *.lp
