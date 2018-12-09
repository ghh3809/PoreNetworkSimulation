global conf;
global batch_i;

saving_name = ['./data/dispersion_', conf.name{batch_i}, '_paths.mat'];
save(saving_name, 'data_file', 'dispersion_type', 'model_size', 'particles',...
    'particles_per_step', 'plot_type', 'res', 'time', 'time_step', 'total_steps', 'unit_size');