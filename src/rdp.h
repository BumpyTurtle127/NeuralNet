// <A> = <B>,<A> | <B>\n<A> | <B>EOF
// <B> = <C> | <C><S> | <S><C>
// <C> = <D> | <D>.<D>
// <D> = <E> | <D><E>
// <E> = 0 | 1 | 2 | ... | 7 | 8 | 9

char line[200];

A(){
	if(B() == 1){
		if(*line == ','){
			line++;
			if(A() == 1) return 1;
		} else if(*line == '\n'){
			line++;
			if(A() == 1) return 1;
		} else if(
	}
}
