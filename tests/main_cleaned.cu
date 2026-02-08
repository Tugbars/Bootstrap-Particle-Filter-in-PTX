
int main(int argc, char** argv) {
    int n_ticks       = 5000;
    int bpf_particles = 50000;
    int apf_particles = 50000;
    int mc_particles  = 10000;
    int base_seed     = 42;
    const char* csv_path = NULL;
    int csv_scenario     = 0;    // 0 = all
    int csv_only         = 0;    // if --csv-only, skip tests
    int use_optimal      = 0;    // if --optimal, use kernel 14

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--ticks") == 0 && i+1 < argc)
            n_ticks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bpf-particles") == 0 && i+1 < argc)
            bpf_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--apf-particles") == 0 && i+1 < argc)
            apf_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mc-particles") == 0 && i+1 < argc)
            mc_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            base_seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i+1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--csv-scenario") == 0 && i+1 < argc)
            csv_scenario = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv-only") == 0)
            csv_only = 1;
        else if (strcmp(argv[i], "--optimal") == 0)
            use_optimal = 1;
    }

    // Enable optimal proposal (kernel 14) if requested — must be before gpu_bpf_create
    if (use_optimal) {
        gpu_bpf_use_optimal_proposal(1);
    }

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  GPU BPF / APF — MATCHED-DGP TEST SUITE\n");
    printf("  Device: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("  Zero model mismatch — all filters use TRUE DGP parameters\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  Config: %d ticks, BPF=%dK, APF=%dK, MC=%dK\n",
           n_ticks, bpf_particles / 1000, apf_particles / 1000,
           mc_particles / 1000);
    printf("  Proposal: %s\n", use_optimal ? "OPTIMAL (kernel 14)" : "bootstrap (kernel 3)");
    if (csv_path)
        printf("  CSV output: %s (scenario=%s)\n",
               csv_path, csv_scenario ? "filtered" : "all");

    // Export CSV if requested (runs BEFORE tests — independent)
    if (csv_path) {
        export_csv(csv_path, n_ticks, bpf_particles, apf_particles,
                   base_seed, csv_scenario);
        if (csv_only) {
            printf("\n--csv-only: skipping test suite.\n");
            return 0;
        }
    }

    // Run all tests
    test_full_comparison(n_ticks, bpf_particles, apf_particles, base_seed);
    test_bpf_particle_sweep(n_ticks, base_seed);
    test_apf_vs_bpf(n_ticks, base_seed);
    test_mc_variance(n_ticks, mc_particles, base_seed);
    test_throughput(base_seed);

    printf("\nAll tests complete.\n");
    return 0;
}
