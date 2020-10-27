#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include<cmath>
#include <quadmath.h>
#include <time.h>
#include <fstream>
#include <sstream>

#define _low -5.0
#define _high 5.0

using namespace std ;

double _r_X ;
double _r_Y ;

template<class T>
void init() 
{
	_r_X = _low + static_cast<T> (rand())/(static_cast<T>(RAND_MAX/(_high - _low))) ;
	_r_Y = _low + static_cast<T> (rand())/(static_cast<T>(RAND_MAX/(_high - _low))) ;
}

template<class T>
T execute_spec_precision(int conditions[])
{


	T r_X = (T) _r_X ;
	T r_Y = (T) _r_Y ;
	T res ;

	if ( r_Y <= 0) {
	conditions[0] = 1;
		if ( r_X <= 0) {
		    conditions[1] = 1;
			res = -1.4717 + 2.83079*r_X + 0.786996*r_X*r_X + 2.83079*r_Y - 1.07939e-16*r_X*r_Y + 0.786996*r_Y*r_Y;
		} else {
		    conditions[1] = 0;
			res = -1.4717 - 2.33079*r_X + 0.786996*r_X*r_X + 2.83079*r_Y + 9.1748e-16*r_X*r_Y + 0.786996*r_Y*r_Y;
		}
	} else {
	    conditions[0] = 0;
		if ( r_X <= 0) {
		    conditions[2] = 1;
			res = -1.4717 + 2.83079*r_X + 0.786996*r_X*r_X - 2.33079*r_Y + 3.23816e-16*r_X*r_Y + 0.786996*r_Y*r_Y;
		} else {
		    conditions[2] = 0;
			res = -1.4717 - 2.33079*r_X + 0.786996*r_X*r_X - 2.33079*r_Y + 1.72702e-15*r_X*r_Y + 0.786996*r_Y*r_Y;
		}
	}

	return res ;
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

int main(int argc, char** argv)

{

	srand(time(0));

	FILE *fp ;
	int N;
	sscanf(argv[1], "%d", &N);
	fp = fopen("styblinski_profile.csv", "w+");
ofstream fp_divergence_inputs;
	fp_divergence_inputs.open("styblinski_divergence_inputs.csv", ios::out | ios::app);

    __float80 val_lp = 0;
	__float80 val_dp = 0;
	__float80 val_qp = 0;
	__float80 err_qp_lp = 0;
	__float80 err_qp_dp = 0;

    __float80 maxerrlp = 0.0;
	__float80 maxerrdp = 0.0 ;

    int num_predicates = 3;
	int conditions_lp[num_predicates];
	int conditions_dp[num_predicates];
	int conditions_qp[num_predicates];

	for(int i=0; i<N; i++) {
        for(int j = 0; j < num_predicates; j++) {
            conditions_lp[j] = -1;
            conditions_dp[j] = -1;
            conditions_qp[j] = -1;
        }
		init<double>();
		__float80 val_lp = (__float80) execute_spec_precision<float>(conditions_lp);
		__float80 val_dp = (__float80) execute_spec_precision<double>(conditions_dp);
		__float80 val_qp = (__float80) execute_spec_precision<__float128>(conditions_qp);

        err_qp_lp += fabs(val_dp - val_lp);
		err_qp_dp += fabs(val_qp - val_dp);

        if ( maxerrlp < fabs(val_dp - val_lp)) maxerrlp = fabs(val_dp - val_lp) ;
		if ( maxerrdp < fabs(val_qp - val_dp)) maxerrdp = fabs(val_qp - val_dp) ;
		for(int j = 0; j < num_predicates; j++) {
            if(conditions_lp[j] != conditions_dp[j] && conditions_lp[j] != -1 && conditions_dp[j] != -1) {
                string str = "instability_lp:" + to_string_with_precision(fabs(val_dp - val_lp), 16) + ",Pred:" + to_string(j) + ",_r_X:" + to_string_with_precision(_r_X, 16) + ",_r_Y:" + to_string_with_precision(_r_Y, 16) + "\n";
                fp_divergence_inputs << str;
                cout << str;
            }
            if(conditions_dp[j] != conditions_qp[j] && conditions_dp[j] != -1 && conditions_qp[j] != -1) {
                string str = "instability_dp:" + to_string_with_precision(fabs(val_qp - val_dp), 16) + ",Pred:" + to_string(j) + ",_r_X:" + to_string_with_precision(_r_X, 16) + ",_r_Y:" + to_string_with_precision(_r_Y, 16) + "\n";
                fp_divergence_inputs << str;
                cout << str;
            }
        }
	}

	fclose(fp);
	fp_divergence_inputs.close();

    cout << "Avg Error in LP -> " << err_qp_lp/N << endl ;
	cout << "Max Error in LP -> " << maxerrlp << endl ;
	cout << "Avg Error in DP -> " << err_qp_dp/N << endl ;
	cout << "Max Error in DP -> " << maxerrdp << endl ;


	return 1;

}
