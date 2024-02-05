package com.aware.phone.ui;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;

import com.aware.Aware;
import com.aware.phone.R;

public class Data extends Aware_Activity{

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.aware_light_main);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String s) {
        // Do nothing
    }


}
