package com.aware.phone.ui;

import static com.aware.utils.PermissionUtils.MANDATORY_PERMISSIONS_GRANTED;
import static com.aware.utils.PermissionUtils.SERVICE_FULL_PERMISSIONS_NOT_GRANTED;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.*;
import android.database.Cursor;
import android.database.DatabaseUtils;
import android.os.AsyncTask;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.*;

import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;
import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.phone.R;
import com.aware.providers.Aware_Provider;
import com.aware.utils.*;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;


public class Configure extends Aware_Activity {

    private static boolean pluginsInstalled = false;
    private boolean isFilledUrl = false;
    private boolean isFilledId = false;
    private Button joinBtn;

    public static final String EXTRA_STUDY_URL = "study_url";
    public static final String EXTRA_STUDY_CONFIG = "study_config";
    public static final String INPUT_PASSWORD = "input_password";

    private RecyclerView permissionsRecyclerView;
    private RecyclerView.Adapter permissionsAdapter;
    private RecyclerView.LayoutManager permissionsLayoutManager;

    private final PermissionUtils.SingleServicePermissionReceiver singleServicePermissionReceiver = new PermissionUtils.SingleServicePermissionReceiver(Configure.this);

    private final MandatoryPermissionGrantedReceiver mandatoryPermissionGrantedReceiver = new MandatoryPermissionGrantedReceiver(Configure.this);

    private static boolean mandatoryPermissionsGranted = false;

    private static boolean isStudyValid = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.configure);
        pluginsInstalled = false;

        //HACK: Unmerged from Aware_Join_Study
//        //If we are getting here from an AWARE study link (deeplink)
//        String scheme = getIntent().getScheme();
//        if (scheme != null) {
//            if (Aware.DEBUG)
//                Log.d(Aware.TAG, "AWARE Link detected: " + getIntent().getDataString() + " SCHEME: " + scheme);
//            if (scheme.equalsIgnoreCase("aware")) {
//                studyUrl = getIntent().getDataString().replace("aware://", "https://");
//            }
//        }

        // Set up list of mandatory permissions
//        ArrayList<PermissionInfo> mandatoryPermissions = new ArrayList<>();
//        permissionsRecyclerView = (RecyclerView) findViewById(R.id.rv_mandatory_permissions);
//        permissionsLayoutManager = new LinearLayoutManager(this);
//        permissionsRecyclerView.setLayoutManager(permissionsLayoutManager);
//
//        PermissionListAdapter listAdapter = new PermissionListAdapter(mandatoryPermissions);
//        permissionsRecyclerView.setAdapter(listAdapter);

        IntentFilter permissionResults = new IntentFilter();
        permissionResults.addAction(SERVICE_FULL_PERMISSIONS_NOT_GRANTED);
        registerReceiver(singleServicePermissionReceiver, permissionResults);

        IntentFilter mandatoryPermissionsResults = new IntentFilter();
        mandatoryPermissionsResults.addAction(MANDATORY_PERMISSIONS_GRANTED);
        registerReceiver(mandatoryPermissionGrantedReceiver, mandatoryPermissionsResults);

        // Listen to changes in URL input
        EditText etStudyConfigUrl = findViewById(R.id.et_join_study_url);
        joinBtn = findViewById(R.id.btn_configure);
        etStudyConfigUrl.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                isFilledUrl = s.length() > 0;
                if (isFilledUrl && isFilledId) {
                    joinBtn.setEnabled(true);
                    joinBtn.setBackgroundColor(ContextCompat.getColor(getApplicationContext(), R.color.accent));
                } else {
                    joinBtn.setEnabled(false);
                    joinBtn.setBackgroundColor(ContextCompat.getColor(getApplicationContext(), R.color.settingDisabled));
                }
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        // Listen to changes in input study identifier
        EditText participant_label = findViewById(R.id.participant_label);
        participant_label.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                Aware.setSetting(getApplicationContext(), Aware_Preferences.DEVICE_LABEL, s.toString());
                isFilledId = s.length() > 0;
                if (isFilledUrl && isFilledId) {
                    joinBtn.setEnabled(true);
                    joinBtn.setBackgroundColor(ContextCompat.getColor(getApplicationContext(), R.color.accent));
                } else {
                    joinBtn.setEnabled(false);
                    joinBtn.setBackgroundColor(ContextCompat.getColor(getApplicationContext(), R.color.settingDisabled));
                }
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        // Validate and retrieve study details when button is clicked
        joinBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                new ValidateURL().execute(etStudyConfigUrl.getText().toString(), "");
            }
        });
    }

    /**
     * Ensures that the main intent is only started after the link is verified and mandatory permissions are granted.
     */
    private class MandatoryPermissionGrantedReceiver extends BroadcastReceiver {
        private Configure currentActivity;

        public MandatoryPermissionGrantedReceiver(Configure activity) {
            this.currentActivity = activity;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            if (!currentActivity.mandatoryPermissionsGranted) {
                currentActivity.mandatoryPermissionsGranted = true;
                if (currentActivity.isStudyValid) {
                    currentActivity.startMainIntent();
                }
            }
        }
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String s) {
        // Do nothing
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterReceiver(singleServicePermissionReceiver);
        unregisterReceiver(mandatoryPermissionGrantedReceiver);
    }

    private void populateStudyDetails(String studyURL, JSONObject studyConfig, String studyPassword, String studyAPI) {
        try {
            Cursor dbStudy = Aware.getStudy(getApplicationContext(), studyURL);
            JSONObject studyInfo = studyConfig.getJSONObject("study_info");

            if (Aware.DEBUG)
                Log.d(Aware.TAG, DatabaseUtils.dumpCursorToString(dbStudy));

            if (dbStudy == null || !dbStudy.moveToFirst()) {
                ContentValues studyData = new ContentValues();
                studyData.put(Aware_Provider.Aware_Studies.STUDY_DEVICE_ID, Aware.getSetting(getApplicationContext(), Aware_Preferences.DEVICE_ID));
                studyData.put(Aware_Provider.Aware_Studies.STUDY_TIMESTAMP, System.currentTimeMillis());
                studyData.put(Aware_Provider.Aware_Studies.STUDY_API, studyAPI);
                studyData.put(Aware_Provider.Aware_Studies.STUDY_URL, studyURL);
                studyData.put(Aware_Provider.Aware_Studies.STUDY_CONFIG, studyConfig.toString());
                studyData.put(Aware_Provider.Aware_Studies.STUDY_KEY, "0"); // studyInfo.getString("id")); STUDY_KEY needs type INT
                studyData.put(Aware_Provider.Aware_Studies.STUDY_PI, studyInfo.getString("researcher_first") + " " + studyInfo.getString("researcher_last") + "\nContact: " + studyInfo.getString("researcher_contact"));
                studyData.put(Aware_Provider.Aware_Studies.STUDY_TITLE, studyInfo.getString("study_title"));
                studyData.put(Aware_Provider.Aware_Studies.STUDY_DESCRIPTION, studyInfo.getString("study_description"));

                getContentResolver().insert(Aware_Provider.Aware_Studies.CONTENT_URI, studyData);

                if (Aware.DEBUG) {
                    Log.d(Aware.TAG, "New study data: " + studyData.toString());
                }
            } else {
                //Update the information to the latest
                ContentValues studyData = new ContentValues();
                studyData.put(Aware_Provider.Aware_Studies.STUDY_DEVICE_ID, Aware.getSetting(getApplicationContext(), Aware_Preferences.DEVICE_ID));
                studyData.put(Aware_Provider.Aware_Studies.STUDY_TIMESTAMP, System.currentTimeMillis());
                studyData.put(Aware_Provider.Aware_Studies.STUDY_JOINED, 0);
                studyData.put(Aware_Provider.Aware_Studies.STUDY_EXIT, 0);
                studyData.put(Aware_Provider.Aware_Studies.STUDY_API, studyAPI);
                studyData.put(Aware_Provider.Aware_Studies.STUDY_URL, studyURL);
                studyData.put(Aware_Provider.Aware_Studies.STUDY_CONFIG, studyConfig.toString());
                studyData.put(Aware_Provider.Aware_Studies.STUDY_KEY, "0"); // studyInfo.getString("id")); STUDY_KEY needs type INT
                studyData.put(Aware_Provider.Aware_Studies.STUDY_PI, studyInfo.getString("researcher_first") + " " + studyInfo.getString("researcher_last") + "\nContact: " + studyInfo.getString("researcher_contact"));
                studyData.put(Aware_Provider.Aware_Studies.STUDY_TITLE, studyInfo.getString("study_title"));
                studyData.put(Aware_Provider.Aware_Studies.STUDY_DESCRIPTION, studyInfo.getString("study_description"));

                getContentResolver().insert(Aware_Provider.Aware_Studies.CONTENT_URI, studyData);

                if (Aware.DEBUG) {
                    Log.d(Aware.TAG, "Re-scanned study data: " + studyData.toString());
                }
            }

            if (dbStudy != null && !dbStudy.isClosed()) dbStudy.close();
            configureSettings(studyConfig, studyURL, studyPassword);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    private void configureSettings(JSONObject studyConfig, String url, String password) {
        ContentValues studyData = new ContentValues();
        studyData.put(Aware_Provider.Aware_Studies.STUDY_JOINED, System.currentTimeMillis());
        studyData.put(Aware_Provider.Aware_Studies.STUDY_EXIT, 0);
        getContentResolver().update(Aware_Provider.Aware_Studies.CONTENT_URI, studyData, Aware_Provider.Aware_Studies.STUDY_URL + " LIKE '" + url + "'", null);

        JSONArray configs = new JSONArray().put(studyConfig);
        StudyUtils.applySettings(getApplicationContext(), url, configs, password);
    }

    /**
     * Triggers to start main intent
     */
    private void startMainIntent() {
        Intent mainUI = new Intent(getApplicationContext(), Aware_Light_Client.class);
        mainUI.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(mainUI);
    }

    private class ValidateURL extends AsyncTask<String, Void, String> {
        private ProgressDialog mLoader;
        private String url;
        private JSONObject studyConfig;
        private String password;
        private Boolean isValidPassword = false;
        private Boolean isValidURL = false;

        @Override
        protected void onPreExecute() {
            mLoader = new ProgressDialog(Configure.this);
            mLoader.setTitle(R.string.loading_join_study_title);
            mLoader.setMessage(getResources().getString(R.string.loading_join_study_msg));
            mLoader.setCancelable(false);
            mLoader.setIndeterminate(true);
            mLoader.show();
        }
        @Override
        protected String doInBackground(String... strings) {
            Log.i(Aware.TAG, "Retrieving study with URL " + url);

            url = strings[0];
            password = strings[1];
            try {
                // 1 - Retrieve configuration from URL
                studyConfig = StudyUtils.getStudyConfig(url);
                if (studyConfig != null){
                    isValidURL = true;
                } else {
                    return null;
                }

                // 2 - Verify if configuration is valid
                isValidPassword = StudyUtils.validateStudyConfig(Configure.this, studyConfig, password);
                if (!isValidPassword) {
                    return null;
                }

                // 3a - Proceed to set up if all inputs are valid
                if (isValidURL && isValidPassword) {
//                    // Set up listener to plugin installation broadcasts
//                    pluginCompliance = new PluginCompliance();
//                    IntentFilter pluginStatuses = new IntentFilter();
//                    pluginStatuses.addAction(Aware.ACTION_AWARE_PLUGIN_INSTALLED);
//                    pluginStatuses.addAction(Aware.ACTION_AWARE_PLUGIN_UNINSTALLED);
//                    registerReceiver(pluginCompliance, pluginStatuses);

                    // Populate study details
                    Cursor qry = Aware.getStudy(getApplicationContext(), url);
                    if (qry == null || !qry.moveToFirst()) {
                        populateStudyDetails(url, studyConfig, "", "");
                    }
                    if (qry != null && !qry.isClosed()) qry.close();

                    // Set up sensor and plugin settings
                    configureSettings(studyConfig, url, password);
                    return studyConfig.toString();
                } else {
                    Log.d(Aware.TAG, "Failed to retrieve study with URL: " + url);
                }
            } catch (JSONException e) {
                Log.d(Aware.TAG, "Failed to retrieve study with URL: " + url + ", reason: " + e.getMessage());
            }
            return null;
        }

        @Override
        protected void onPostExecute(String studyConfig) {
            mLoader.dismiss();
            android.app.AlertDialog.Builder builder;
            if (studyConfig == null) {
                builder = new android.app.AlertDialog.Builder(Configure.this);
                builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        setResult(Activity.RESULT_CANCELED);
                        //Reset the webservice server status
                        Aware.setSetting(getApplicationContext(), Aware_Preferences.STATUS_WEBSERVICE, false);

                        dialog.dismiss();
                    }
                });
                builder.setMessage("Failed to retrieve study information. Please try again.");
                if (!isValidURL) {
                    builder.setTitle("Invalid URL");
                }
                else if (!isValidPassword) {
                    builder.setTitle("Incorrect password");
                }
                builder.show();
            } else {
                isStudyValid = true;
                if (mandatoryPermissionsGranted) {
                    startMainIntent();
                }
            }
        }
    }
}
