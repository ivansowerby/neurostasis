#real time LSL streaming using EEG headset
import gpype as gp

fs = 250

if __name__ == "__main__":
    #app = gp.MainApp()
    p = gp.Pipeline()

    # 1. HARDWARE SOURCE
    source = gp.HybridBlack(
        include_accel=True, 
        include_gyro=True,  
        include_aux=True,   
    )

    # 2. KEYBOARD FOR EVENT MARKERS (Added from LSL example)
    #keyboard = gp.Keyboard()

    # 3. FILTERING CHAIN (Cleaning the EEG)
    splitter = gp.Router(input_channels=gp.Router.ALL,
                         output_channels={"EEG": range(8),
                                          "ACC": [8, 9, 10],
                                          "GYRO": [11, 12, 13],
                                          "AUX": [14, 15, 16]})
    
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)
    notch60 = gp.Bandstop(f_lo=58, f_hi=62)

    # 4. DATA PACKAGING (Merging Filtered EEG + Hardware Sensors + Keyboard)
    # This creates a single wide stream for LSL
    router = gp.Router(input_channels=[gp.Router.ALL])
    
    # 5. LSL SENDER (Network Output)
    sender = gp.LSLSender(stream_name="Unicorn_Hybrid_Black", stype="EEG")

    # 6. VISUALIZATION & STORAGE
    #scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)
    #writer = gp.CsvWriter(file_name="eeg_data.csv")

    # === PIPELINE CONNECTIONS ===

    
    # Connect processing chain
    p.connect(source, splitter)
    p.connect(splitter["EEG"], bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50, notch60)

    # CRITICAL FIX: Connect the filtered data to the LSL sender
    p.connect(notch60, sender)

    # The "Bundle": 
    # Input 1: Clean EEG + ACC + GYRO + AUX (Total 17 channels)
    # Input 2: Keyboard Events (1 channel)
    # Total LSL Channels = 18
    #p.connect(source, lsl_combiner["in1"]) 
    #p.connect(keyboard, lsl_combiner["in2"])
    
    # Send bundled data to LSL
    #p.connect(lsl_combiner, sender)
    #p.connect(source, sender)

    # Keep original visual/save path
    #p.connect(notch60, scope)
    #p.connect(source, writer)

    # === START ===

    p.start()
    input("Streaming to LSL: 'Unicorn_Hybrid_Black_Stream', press Enter to stop...\n")
    p.stop()
