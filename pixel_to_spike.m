% the function coverts pixel input vector (0 - 255) to spike train of specified duration


function spike_train = pixel_to_spike(input, dt, duration, max_rate)
    
    rescale_factor = 1/(dt * max_rate);        
    temp = rand(size(input,1), size(input,2), floor(duration/dt)) * rescale_factor;    
    spike_train = temp <= input;    
        
end