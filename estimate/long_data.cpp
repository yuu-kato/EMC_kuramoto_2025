// Framework:
// To create long time series data generated from Kuramoto model
// oscillator number = 100, 1000, 100000
// the seed is determined based on the results of repeated EMC simulations

#include<bits/stdc++.h>
#include<sys/stat.h>
#include<sys/types.h>
#include<omp.h>
using namespace std;
double pi = acos(-1);

// input : t, output : R
// parameter : K, gamma

// create Lorentz distribution
void generate_omega(double mu, double gamma_true, int M, vector<double> &omega, int seed) {
    //random generator for data generation
    mt19937 gen_data(0 + seed); //fix seed for the creation of natural frequencies
    cauchy_distribution<float> cauchy(mu, gamma_true);

    for (int i = 0; i < M; i++){
        omega.at(i) = cauchy(gen_data);
    }
    return ;
}

// Kuramoto model written with mean field (r)
// definition: phi denotes the phases of the oscillators
vector<double> kuramoto_dfdt(const vector<double> &phi, const vector<double> &omega, double K_true, const vector<double>& r) {
    int M = phi.size();
    vector<double> dphidt(M);

    #pragma omp parallel for
    for (int i = 0; i < M; i++){
        dphidt.at(i) = omega.at(i) + K_true * r.at(0) * sin(r.at(1) - phi.at(i));
    }
    return dphidt;
}

// calculate order parameter from phi
vector<double> orderparam(const vector<double> &phi) {
    int M = phi.size();
    double rx = 0.0, ry = 0.0;

    // summing cos(phi) and sin(phi) over all oscillators
    #pragma omp parallel for reduction(+:rx, ry)
    for (int i = 0; i < M; i++) {
        rx += cos(phi[i]);
        ry += sin(phi[i]);
    }

    rx /= M;
    ry /= M;

    double r = sqrt(rx * rx + ry * ry);
    double phi_r = atan2(ry, rx);

    return {r, phi_r};
}

// artificial data
void data_generate_kuramoto(vector<double> &t_data, vector<double> &R_data, double t_max, double dt, double K_true, double gamma_true, double mu, int M, int N, int seed){
    // generate natural frequencies
    vector<double> omega(M);
    generate_omega(mu, gamma_true, M, omega, seed);

    // initial condition
    vector<double> phi(M, 0.0); //ic to all 0
    t_data.at(0) = 0.0;
    R_data.at(0) = orderparam(phi).at(0);

    // 4th order Runge-Kutta method
    int N_rk = t_max/dt + 1;
    vector<double> k1(M), k2(M), k3(M), k4(M), phi_tmp(M);
    for (int i = 1; i < N_rk; i++) {

        // k1
        k1 = kuramoto_dfdt(phi, omega, K_true, orderparam(phi));

        // k2
        #pragma omp parallel for
        for (int j = 0; j < M; j++) {
            phi_tmp[j] = phi[j] + 0.5 * dt * k1[j];
        }
        k2 = kuramoto_dfdt(phi_tmp, omega, K_true, orderparam(phi_tmp));

        // k3
        #pragma omp parallel for
        for (int j = 0; j < M; j++) {
            phi_tmp[j] = phi[j] + 0.5 * dt * k2[j];
        }
        k3 = kuramoto_dfdt(phi_tmp, omega, K_true, orderparam(phi_tmp));

        // k4
        #pragma omp parallel for
        for (int j = 0; j < M; j++) {
            phi_tmp[j] = phi[j] + dt * k3[j];
        }
        k4 = kuramoto_dfdt(phi_tmp, omega, K_true, orderparam(phi_tmp));

        // update phi
        #pragma omp parallel for
        for (int j = 0; j < M; j++) {
            phi[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }

        int index = i / ((N_rk - 1)/(N-1));

        if (i % ((N_rk - 1)/(N-1)) == 0){
            t_data.at(index) = i * dt;
            R_data.at(index) = orderparam(phi).at(0);
        }
    }
    return ;
}

int main() {
    //simulation start
    struct timespec startTime, endTime;//time
    clock_gettime(CLOCK_REALTIME, &startTime);

    // output files
    mkdir("./output_files_long_data", 0755); // 0755 means 'allowing the owner to write, read, execute, while allowing others only to read and execute'
    ofstream Data_out0("./output_files_long_data/true_parameters.csv");
    ofstream Data_out1("./output_files_long_data/data.csv");
    Data_out1 << "oscillator_number" << "," << "seed" << ",";

    ////////// PARAMETERS //////////
    //// system parameters ////
    double gamma_true = 0.08;
    double K_true = 0.05;
    double mu = 0.0;

    //// parameters for data creation ////
    int M; // number of oscillators, defined later
    int N = 20001; // number of data points
    double t_max = 10000.0; // max time
    double dt = 0.001; 
    int seed; // seed of random number, defined later

    // record parameters 
    Data_out0 << "gamma_true" << "," << "K_true" <<  "," << "mu" << "," << "t_max" << "," <<  "N" << "," << "dt" << endl;

    Data_out0 << gamma_true << "," << K_true << "," << mu << "," << t_max << "," << N << "," << dt  << endl;
    /////////////////////////////

    // list of oscillator numbers and seeds
    vector<int> M_list = {100, 1000, 100000};
    vector<int> seed_list = {1459, 2419, 4032};

    for (int NUM=0; NUM < 3; NUM++){
        M = M_list.at(NUM);
        seed = seed_list.at(NUM);

        // generate synthetic data
        vector<double> t_data(N, 0.0);
        vector<double> R_data(N, 0.0);
        data_generate_kuramoto(t_data, R_data, t_max, dt, K_true, gamma_true, mu, M, N, seed);

        // record data
        if (NUM == 0) {
            for (int n = 0; n < (N-1); n++){
                Data_out1 << t_data.at(n) << ",";
            }
            Data_out1 << t_data.at(N-1) << endl;
        }
        Data_out1 << M << "," << seed << ",";
        for (int n = 0; n < (N-1); n++){
            Data_out1 << R_data.at(n) << ",";
        }
        Data_out1 << R_data.at(N-1) << endl;
    }

    clock_gettime(CLOCK_REALTIME, &endTime);
    printf("elapsed time = ");
    if (endTime.tv_nsec < startTime.tv_nsec) {
    printf("%5ld.%09ld", endTime.tv_sec - startTime.tv_sec - 1,
            endTime.tv_nsec + (long int)1.0e+9 - startTime.tv_nsec);
    } 
    else {
    printf("%5ld.%09ld", endTime.tv_sec - startTime.tv_sec,
            endTime.tv_nsec - startTime.tv_nsec);
    }
    printf("(sec)\n");
}


