package com.aware.phone.ui;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import com.aware.Aware;
import com.aware.phone.R;

/**
 * Main entry point that directs to the sign up or home page.
 */
public class Landing_Page extends Aware_Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent activityToRedirect = null;
        if (Aware.isStudy(getApplicationContext())) {
            activityToRedirect = new Intent(getApplicationContext(), Aware_Light_Client.class);
        } else {
            activityToRedirect = new Intent(getApplicationContext(), Configure.class);
        }
        if (activityToRedirect != null) {
            startActivity(activityToRedirect);
            finish();
        }
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String s) {
        // Do nothing
    }
}
