package com.aware.phone.ui;

import android.Manifest;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import androidx.annotation.NonNull;
import androidx.core.content.PermissionChecker;
import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.phone.R;
import com.aware.phone.ui.dialogs.JoinStudyDialog;
import com.aware.ui.PermissionsHandler;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import java.util.ArrayList;

public abstract class Aware_Activity extends AppCompatPreferenceActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(false);
            getSupportActionBar().setDisplayShowHomeEnabled(false);
        }

        BottomNavigationView bottomNavigationView = (BottomNavigationView) findViewById(R.id.aware_bottombar);

        // HACK: Retrieves intent extra to change navigation drawer background
        if (getIntent() != null && getIntent().getExtras() != null) {
            int pageId = getIntent().getIntExtra("page", 0);
            Menu menu = bottomNavigationView.getMenu();
            for (int i = 0; i < menu.size(); i++) {
                MenuItem item = menu.getItem(i);
                item.setChecked(false);
            }
            MenuItem selectedItem = menu.getItem(pageId);
            selectedItem.setChecked(true);
        }


        if (bottomNavigationView != null) {
            bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
                @Override
                public boolean onNavigationItemSelected(@NonNull MenuItem item) {
                    //HACK: Include menu item id in intent to highlight item background in navigation drawer
                    switch (item.getItemId()) {
                        case R.id.home: //Home
                            Intent mainUI = new Intent(getApplicationContext(), Aware_Light_Client.class);
                            mainUI.putExtra("page", 0);
                            startActivity(mainUI);
                            break;
                        case R.id.settings: //Settings
                            // Intent pluginsManager = new Intent(getApplicationContext(), Plugins_Manager.class);
                            Intent settingsUI = new Intent(getApplicationContext(), Settings_Page.class);
                            settingsUI.putExtra("page", 1);
                            startActivity(settingsUI);
                            break;
                        case R.id.data: //Stream
                            Intent dataUI = new Intent(getApplicationContext(), Stream_UI.class);
                            dataUI.putExtra("page", 2);
                            startActivity(dataUI);
                            break;
                    }
                    return true;
                }
            });
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.aware_menu, menu);
        for (int i = 0; i < menu.size(); i++) {
            MenuItem item = menu.getItem(i);
            if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_qrcode)) && Aware.is_watch(this))
                item.setVisible(false);
//            if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_team)) && Aware.is_watch(this))
//                item.setVisible(false);
            if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_study)) && Aware.is_watch(this))
                item.setVisible(false);
//            if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_sync)) && !Aware.isStudy(this))
//                item.setVisible(false);
            if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_study)) && !Aware.isStudy(this))
                item.setVisible(false);
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_qrcode))) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && PermissionChecker.checkSelfPermission(this, Manifest.permission.CAMERA) != PermissionChecker.PERMISSION_GRANTED) {
                ArrayList<String> permission = new ArrayList<>();
                permission.add(Manifest.permission.CAMERA);

                Intent permissions = new Intent(this, PermissionsHandler.class);
                permissions.putExtra(PermissionsHandler.EXTRA_REQUIRED_PERMISSIONS, permission);
                permissions.putExtra(PermissionsHandler.EXTRA_REDIRECT_ACTIVITY, getPackageName() + "/" + getPackageName() + ".ui.Aware_QRCode");
                permissions.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                startActivity(permissions);
            } else {
                Intent qrcode = new Intent(Aware_Activity.this, Aware_QRCode.class);
                qrcode.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
                startActivity(qrcode);
            }
        }
        if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_study))) {
            Intent studyInfo = new Intent(Aware_Activity.this, Aware_Join_Study.class);
            studyInfo.putExtra(Aware_Join_Study.EXTRA_STUDY_URL, Aware.getSetting(this, Aware_Preferences.WEBSERVICE_SERVER));
            startActivity(studyInfo);
        }
//        if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_team))) {
//            Intent about_us = new Intent(Aware_Activity.this, About.class);
//            startActivity(about_us);
//        }
//        if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_sync))) {
//            Toast.makeText(getApplicationContext(), "Syncing data...", Toast.LENGTH_SHORT).show();
//            Intent sync = new Intent(Aware.ACTION_AWARE_SYNC_DATA);
//            sendBroadcast(sync);
//        }
        if (item.getTitle().toString().equalsIgnoreCase(getResources().getString(R.string.aware_join_study_link))) {
            new JoinStudyDialog(Aware_Activity.this).showDialog();
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onResume() {
        super.onResume();
        setTitle("AWARE-Light");
    }
}
