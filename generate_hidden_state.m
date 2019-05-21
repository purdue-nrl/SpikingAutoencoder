function imdb = generate_hidden_state(imdb, ae, opts)

input = pixel_to_spike(imdb.image, opts.dt, opts.duration, opts.max_rate);
imdb.hidden_state = zeros(ae.hidden_size, size(imdb.image,2), opts.T);
opts.batch_size = size(imdb.image, 2);
ae = ae.initialize(opts);


for t = 1:opts.T
   ae = ae.code(input(:,:,t), opts);
   imdb.hidden_state(:,:,t) = ae.hidden.spikes;
end
        

end