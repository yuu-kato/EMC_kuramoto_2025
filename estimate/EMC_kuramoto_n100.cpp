// Framework:
// Data generated from Kuramoto model
// Single estimation when oscillator number = 100
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

// analytical sulution by Ott-Antonsen ansatz
double R_t(double t, double K, double gamma){
    double R;
    double R_0 = 1.0;
    R = exp(-gamma * t + 0.5 * K * t) / sqrt(1.0/R_0/R_0 + (exp(-2.0 * gamma * t + K * t) - 1.0) * K / (-2*gamma + K)); 
    return R;
}

// logarithm of prior distribution (alpha < beta)
// definition of parameter : theta := {gamma, delta_gamma}, where delta_gamma = gamma - K/2.
    // assumption: gamma > K/2 
double prior_dist(const vector<double> &theta, double alpha_gamma, double beta_gamma, double alpha_delta_gamma, double beta_delta_gamma) {
    double log_prior = 0.0;
    if (theta.at(0) <= alpha_gamma || theta.at(0) >= beta_gamma) {
        return -1e300;
    }
    log_prior += log(1.0 / (beta_gamma - alpha_gamma));

    if (theta.at(1) <= alpha_delta_gamma || theta.at(1) >= beta_delta_gamma) {
        return -1e300;
    }
    log_prior += log(1.0 / (beta_delta_gamma - alpha_delta_gamma));

    return log_prior;
}

// error function
double error_func(const int N, const vector<double> &t_data, const vector<double> &R_data, const vector<double> &theta) {
    vector<double> f(N);
    double sum = 0.0;
    double gamma = theta.at(0);
    double K = 2*(theta.at(0) - theta.at(1));
    for (int i = 0; i < N; i++) {
        f.at(i) = R_t(t_data.at(i), K, gamma);
        sum += (f.at(i) - R_data.at(i)) * (f.at(i) - R_data.at(i));
    }
    return 0.5 * sum / N;
}

// let b denote the inverse tempreture
double posterior_density(double b, int N, const vector<double> &t_data, const vector<double> &R_data, const vector<double> &theta, double alpha_gamma, double beta_gamma, double alpha_delta_gamma, double beta_delta_gamma) {
    double E = error_func(N, t_data, R_data, theta);
    double ln_pri = prior_dist(theta, alpha_gamma, beta_gamma, alpha_delta_gamma, beta_delta_gamma);
    double post = -b * N * E  + ln_pri;
    return post;
}

// metropolis step
void metropolis(double b, int N, const vector<double> &t_data, const vector<double> &R_data, vector<double> &theta, const vector<double> &s, vector<double> &count, vector<double> &accept_count, int rand_num_int, const vector<double> &rand_num_double, double alpha_gamma, double beta_gamma, double alpha_delta_gamma, double beta_delta_gamma) {  
    int r = rand_num_int;
    double ran_norm = rand_num_double.at(0);
    vector<double> theta_proposed;
    theta_proposed = theta;
    theta_proposed.at(r) += ran_norm * s.at(r);
    count.at(r) += 1.0; 
    
    double R = rand_num_double.at(1);

    if ( log(R) <  prior_dist(theta_proposed, alpha_gamma, beta_gamma, alpha_delta_gamma, beta_delta_gamma) - prior_dist(theta, alpha_gamma, beta_gamma, alpha_delta_gamma, beta_delta_gamma) -b * N * (error_func(N, t_data, R_data, theta_proposed) - error_func(N, t_data, R_data, theta))) {
        theta = theta_proposed;
        accept_count.at(r) += 1.0; 
        return ;
    }
    else {
        return ;
    }
}

// update step size (s)
void update_step(vector<double> &s, const vector<double> &count, const vector<double> &accept_count, double target_accept, double range_accept){
    int num_params = s.size();

    for (int param = 0; param < num_params; param++) {
        double accept_rate = accept_count.at(param) / count.at(param);

        if (abs(accept_rate - target_accept) > range_accept) {
            s.at(param) *= (1 + (accept_rate - target_accept));
        }
    }

    return ;
}

// calculate free energy (used in EMC())
void calculate_free_energy(const vector<double> &b, int &l_index, vector<double> &F_burn, const vector<vector<double>> &error_mat, int N, int L){
    int after_burn = error_mat.at(0).size();

    // set free energy of first replica
    F_burn.at(0) = 5000; //infty

    // find the minimum of free energy
    vector<double> E_burn(after_burn, 0.0);
    vector<double> Q(after_burn, 0.0);
    double F_burn_min = F_burn.at(0);
    double logsumexp = 0.0;
    for (int l = 0; l < L-1; l++){
        double Qmax = - 100000000.0;
        double Q_exp_avr = 0.0;

        for (int i = 0; i < 5000; i++){
            E_burn.at(i) = error_mat.at(l).at(i);
            Q.at(i) = - N * (b.at(l+1) - b.at(l)) * E_burn.at(i);
            if (Qmax < Q.at(i)){
                Qmax = Q.at(i);
            }
        }

        for (int i = 0; i < 5000; i++){
            Q_exp_avr += exp(Q.at(i) - Qmax)/5000;
        }
        logsumexp += Qmax + log(Q_exp_avr);
        F_burn.at(l+1) = -logsumexp - 0.5 * N * log(b.at(l+1)) + 0.5 * N * log(2*pi);

        if (F_burn.at(l+1) < F_burn_min){
            F_burn_min = F_burn.at(l+1);
            l_index = l+1;
        }
    }
    return; 
}

//echange Monte Carlo
void EMC(int L, const vector<double> &b, int N, const vector<double> &t_data, const vector<double> &R_data, vector<vector<double>> s, int simulation_size, int burn_in, double target_accept, double range_accept, int seed, int exchange_freq, double alpha_gamma, double beta_gamma, double alpha_delta_gamma, double beta_delta_gamma){
    //setting of files
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ofstream fout_1("./output_files_kuramoto_n100/parameter.csv");
    fout_1 << "mc_step" << ",";
    for (int l = 0; l < L; l++){
        fout_1 << "l = " + to_string(l) + "_gamma" << "," << "l = " + to_string(l) + "_delta_gamma" << ",";
    }
    fout_1 << endl;

    ofstream fout_2("./output_files_kuramoto_n100/error.csv");
    fout_2 << "mc_step" << ",";
    for (int l = 0; l < L; l++){
        fout_2 << "l = " + to_string(l) + "error" << ",";
    }
    fout_2 << endl;

    ofstream fout_3("./output_files_kuramoto_n100/exchange_rate.csv");
    ofstream fout_4("./output_files_kuramoto_n100/acceptance_rate.csv");
    ofstream fout_5("./output_files_kuramoto_n100/step_size.csv");
    fout_3 << "mc_step" << ",";
    fout_4 << "mc_step" << ",";
    fout_5 << "mc_step" << ",";
    for (int l = 0; l < L; l++) {
        if (l != L-1) {
            fout_3 << to_string(l) + "&" + to_string(l + 1) << ",";
        }
        fout_4 << "l = " + to_string(l) + "_K" << "," << "l = " + to_string(l) + "_gamma" << ",";
        fout_5 << "l = " + to_string(l) + "_K" << "," << "l = " + to_string(l) + "_gamma" << ",";
    }
    fout_3 << endl;
    fout_4 << endl;
    fout_5 << endl;

    ofstream fout_6("./output_files_kuramoto_n100/MAP.csv");
    fout_6 << "l" << ",";
    for (int i = 0; i < L; i++) {
        fout_6 << i << ",";
    }
    fout_6 << endl;

    ofstream fout_7("./output_files_kuramoto_n100/free_energy.txt");
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // random generator for estimation
    mt19937 gen_est(10000 + seed);

    // probability distribution for metropolis
    uniform_int_distribution<> dist_int(0, 1);
    normal_distribution<float> dist_norm_metro(0, 1);
    uniform_real_distribution<float> dist_uni_metro(0, 1);

    // probability distribution for exchange
    uniform_real_distribution<float> dist_exchange(0, 1);

    // parameter, error, posterior
    vector<vector<double>> theta(L ,vector<double>(2)); 
    vector<vector<double>> theta_MAP(L ,vector<double>(2)); 
    vector<double> error(L);
    vector<double> posterior(L);

    // initialize by prior dist
    uniform_real_distribution<float> dist_gamma(alpha_gamma, beta_gamma);
    uniform_real_distribution<float> dist_delta_gamma(alpha_delta_gamma, beta_delta_gamma);
    for (int i = 0; i < L; i++) {
        theta.at(i).at(0) = dist_gamma(gen_est);
        theta.at(i).at(1) = dist_delta_gamma(gen_est);
        error.at(i) = error_func(N, t_data, R_data, theta.at(i));
        posterior.at(i) = posterior_density(b.at(i), N, t_data, R_data, theta.at(i), alpha_gamma, beta_gamma, alpha_delta_gamma, beta_delta_gamma);
    }
    theta_MAP = theta;

    // for counting acceptance ratio
    vector<vector<double>> count(L ,vector<double>(2, 0.0));
    vector<vector<double>> accept_count(L ,vector<double>(2, 0.0));
    vector<double> num_ex(L - 1, 0.0);
    vector<double> count_ex(L - 1, 0.0);

    // for calculating free energy
    int after_burn = (simulation_size - burn_in)/10;
    vector<vector<double>> error_mat(L ,vector<double>(after_burn)); 

    ///////////////////////////////////////////////////////////////
    for (int MC_step = 1; MC_step <= simulation_size ; MC_step++){
        // generate random numbers for metropolis steps
        vector<int> rand_num_int(L, 0);
        vector<vector<double>> rand_num_double(L, vector<double>(2, 0.0));
        for (int j = 0; j < L; j++) {
            rand_num_int.at(j) = dist_int(gen_est);
            rand_num_double.at(j) = {dist_norm_metro(gen_est), dist_uni_metro(gen_est)};
        }

        // parallelization
        double posteri = 0.0;
        #pragma omp parallel for firstprivate(posteri)
        for (int replica_index = 0; replica_index < L; replica_index ++) {
            metropolis(b.at(replica_index), N, t_data, R_data, theta.at(replica_index), s.at(replica_index), count.at(replica_index), accept_count.at(replica_index), rand_num_int.at(replica_index), rand_num_double.at(replica_index), alpha_gamma, beta_gamma, alpha_delta_gamma, beta_delta_gamma);
            error.at(replica_index) = error_func(N, t_data, R_data, theta.at(replica_index));
            posteri = posterior_density(b.at(replica_index), N, t_data, R_data, theta.at(replica_index), alpha_gamma, beta_gamma, alpha_delta_gamma, beta_delta_gamma);
            if (posteri > posterior.at(replica_index)){
                posterior.at(replica_index) = posteri;
                theta_MAP.at(replica_index) = theta.at(replica_index);
            }
        }
        #pragma omp barrier
        //////////////////////////////////////////////////////////////////////

        if (MC_step % 10 == 0){
            fout_1 << MC_step << ",";
            fout_2 << MC_step << ",";
            for (int l = 0; l < L; l++){
                fout_1 << theta.at(l).at(0) << "," << theta.at(l).at(1) << ",";
                fout_2 << error.at(l) << ",";
                if (MC_step > burn_in){
                    error_mat.at(l).at((MC_step - burn_in)/10 - 1) = error.at(l);
                }
            }
            fout_1 << endl;
            fout_2 << endl;
        }

        // exchange
        if (MC_step % exchange_freq == 0) {
            for (int replica_index = 0; replica_index < L - 1; replica_index++) {
                num_ex.at(replica_index) += 1.0;
                double v = N * (b.at(replica_index + 1) - b.at(replica_index)) * (error.at(replica_index + 1) - error.at(replica_index));
                double ran = dist_exchange(gen_est);
                if (log(ran) < v) {
                    count_ex.at(replica_index) += 1.0;
                    swap(theta.at(replica_index), theta.at(replica_index + 1));
                }
            }
        }

        // calculate the acceptance ratio
        if (MC_step % 200 == 0){
            fout_3 << MC_step << ",";
            fout_4 << MC_step << ",";
            fout_5 << MC_step << ",";
            for (int l = 0; l < L; l++) {
                if (l != L-1) {
                    fout_3 << count_ex.at(l) / num_ex.at(l) << ",";
                    num_ex.at(l) == 0.0;
                    count_ex.at(l) == 0.0;
                }
                fout_4 << accept_count.at(l).at(0) / count.at(l).at(0) << "," << accept_count.at(l).at(1) / count.at(l).at(1) << ",";

                // update step size during burn-in period
                if (MC_step <= burn_in){
                    update_step(s.at(l), count.at(l), accept_count.at(l), target_accept, range_accept);
                }
                fout_5 << s.at(l).at(0) << "," << s.at(l).at(1) << ",";

                // reset acceptance rate
                accept_count.at(l).at(0) = 0.0;
                accept_count.at(l).at(1) = 0.0;
                count.at(l).at(0) = 0.0;
                count.at(l).at(1) = 0.0;
            }
            fout_3 << endl;
            fout_4 << endl;
            fout_5 << endl;
        }
    }

    // MAP
    for (int i = 0; i < 2; i++){
        fout_6 << "param_" + to_string(i) << ",";
        for (int l = 0; l < L; l++){
            fout_6 << theta_MAP.at(l).at(i) << ",";
        }
        fout_6 << endl;
    }

    // calculate and record free energy
    int l_index = 0;
    vector<double> F_burn(L, 0.0);
    calculate_free_energy(b, l_index, F_burn, error_mat, N, L);
    for (int l = 0; l < L; l++){
        fout_7 << b.at(l) << "  " << F_burn.at(l) << endl;
    }
}

int main() {
    //simulation start
    struct timespec startTime, endTime;//time
    clock_gettime(CLOCK_REALTIME, &startTime);

    // output files
    mkdir("./output_files_kuramoto_n100", 0755); // 0755 means 'allowing the owner to write, read, execute, while allowing others only to read and execute'
    ofstream fout_1("./output_files_kuramoto_n100/data.txt");
    ofstream fout_2("./output_files_kuramoto_n100/true_parameters.csv");

    ////////// PARAMETERS //////////
    //// system parameters ////
    double gamma_true = 0.08;
    double K_true = 0.05;
    double mu = 0.0;

    //// parameters for data creation ////
    int M = 100; // number of oscillators
    int N = 101; // number of data points
    double t_max = 50.0; // max time
    double dt = 0.001; 
    int seed = 1459; // seed of random number

    //// parameters for estimation ////
    int L_rep = 50; // the number of replicas
    double b_max = 1500000.0;
    double b_ratio = 1.5;

    // for prior distribution
    double alpha_gamma = 0.0;
    double beta_gamma = 1.0;
    double alpha_delta_gamma = 0.0;
    double beta_delta_gamma = 1.0;

    // for step size (STANDARD DEVIATION of normal distribution)
    double C_gamma = 0.01; // determines initial values of step size for gamma 
    double C_delta_gamma = 0.01; // determines initial values of step size for delta_gamma
    double d_gamma = 1.0;
    double d_delta_gamma = 1.0;

    // for EMC
    int exchange_freq = 2;
    int simulation_size = 100000; //total number of Monte Carlo steps
    int burn_in = 50000; // first part of MC steps
    double target_accept = 0.6; //target of acceptance rate
    double range_accept = 0.05; //tolerance range of acceptance rate

    // record parameters 
    fout_2 << "gamma_true" << "," << "K_true" <<  "," << "mu" << "," << "t_max" << "," << "M" << "," << "N" << "," << "dt" << "," << "L_rep" << "," << "b_max" << "," << "b_ratio" << "," << "alpha_gamma" << "," << "beta_gamma" << "," << "alpha_delta_gamma" << "," << "beta_delta_gamma" << "," << "C_gamma" << "," << "C_delta_gamma" << "," << "d_gamma" << "," << "d_delta_gamma" << "," << "exchange_freq" << "," << "simulation_size"  << "," << "burn_in" << "," << "targer_accept" << "," << "range_accept" << endl;

    fout_2 << gamma_true << "," << K_true << "," << mu << "," << t_max << "," << M << "," << N << "," << dt << "," << L_rep << "," << b_max << "," << b_ratio << "," << alpha_gamma << "," << beta_gamma << "," << alpha_delta_gamma << "," << beta_delta_gamma << "," << C_gamma << "," << C_delta_gamma << "," << d_gamma << "," << d_delta_gamma << "," << exchange_freq << "," << simulation_size << "," << burn_in << "," << target_accept << "," << range_accept << endl;
    /////////////////////////////

    // generate synthetic data
    vector<double> t_data(N, 0.0);
    vector<double> R_data(N, 0.0);
    data_generate_kuramoto(t_data, R_data, t_max, dt, K_true, gamma_true, mu, M, N, seed);
    for (int i = 0; i < N; i++){
        fout_1 << t_data.at(i) << "  " << R_data.at(i) << endl;
    }

    // prepare inverse tempretures
    vector<double> b(L_rep);
    for (int i = 0; i < L_rep; i++) {
        b.at(i) = b_max * pow(b_ratio, i - L_rep + 1);
    }
    b.at(0) = 0.0;

    // create initial step size for metropolis sampling
    vector<vector<double>> s(L_rep ,vector<double>(2));
    for (int i = 0; i < L_rep; i++) {
        if (N * b.at(i) < b_max) {
            s.at(i).at(0) = C_gamma;
            s.at(i).at(1) = C_delta_gamma;
        }
        else {
            s.at(i).at(0) = C_gamma /pow(N * (b.at(i)/ b_max), d_gamma);
            s.at(i).at(1) = C_delta_gamma /pow(N* (b.at(i)/ b_max), d_delta_gamma);
        }
    }

    // Estimation by EMC
    EMC(L_rep, b, N, t_data, R_data, s, simulation_size, burn_in, target_accept, range_accept, seed, exchange_freq, alpha_gamma, beta_gamma, alpha_delta_gamma, beta_delta_gamma);

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


