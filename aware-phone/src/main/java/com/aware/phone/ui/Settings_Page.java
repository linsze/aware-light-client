package com.aware.phone.ui;

import static com.aware.Aware.TAG;

import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
import android.graphics.drawable.Drawable;
import android.os.AsyncTask;
import android.os.Bundle;
import android.content.SharedPreferences;
import android.preference.CheckBoxPreference;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.preference.Preference;
import android.preference.PreferenceCategory;
import android.preference.PreferenceScreen;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.ListAdapter;
import android.widget.Toast;

import androidx.appcompat.widget.Toolbar;
import androidx.core.content.ContextCompat;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.phone.R;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Map;

/**
 * Settings page to view and update study, sensors, and plugin configurations.
 */
public class Settings_Page extends Aware_Activity{

    private static SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("com.aware.phone", Context.MODE_PRIVATE);

        setContentView(R.layout.aware_light_main);
        addPreferencesFromResource(R.xml.pref_aware_light);
    }

    @Override
    public boolean onPreferenceTreeClick(PreferenceScreen preferenceScreen, final Preference preference) {
        if (preference instanceof PreferenceScreen) {
            Dialog subpref = ((PreferenceScreen) preference).getDialog();

            //HACK: Inflate plugin preferences into the current layout.
            if (subpref == null) {
                String prefKey = preference.getKey();
                Class pluginClass = null;
                switch (prefKey) {
                    case "plugin_activity_recognition":
                        pluginClass = com.aware.plugin.google.activity_recognition.Settings.class;
                        break;
                    case "plugin_ambient_noise":
                        pluginClass = com.aware.plugin.ambient_noise.Settings.class;
                        break;
                    case "plugin_device_usage":
                        pluginClass = com.aware.plugin.device_usage.Settings.class;
                        break;
                }
                if (pluginClass != null) {
                    Intent pluginPrefIntent = new Intent(this, pluginClass);
                    startActivity(pluginPrefIntent);
                }
                return true;
            }

            ViewGroup root = (ViewGroup) subpref.findViewById(android.R.id.content).getParent();
            Toolbar toolbar = new Toolbar(this);
            toolbar.setBackgroundColor(ContextCompat.getColor(preferenceScreen.getContext(), R.color.primary));
            toolbar.setTitleTextColor(ContextCompat.getColor(preferenceScreen.getContext(), android.R.color.white));
            toolbar.setTitle(preference.getTitle());
            root.addView(toolbar, 0); //add to the top

            subpref.setOnDismissListener(new DialogInterface.OnDismissListener() {
                @Override
                public void onDismiss(DialogInterface dialog) {
                    new SettingsSync().execute(preference);
                }
            });
        }
        return super.onPreferenceTreeClick(preferenceScreen, preference);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        String value = "";
        Map<String, ?> keys = sharedPreferences.getAll();
        if (keys.containsKey(key)) {
            Object entry = keys.get(key);
            if (entry instanceof Boolean)
                value = String.valueOf(sharedPreferences.getBoolean(key, false));
            else if (entry instanceof String)
                value = String.valueOf(sharedPreferences.getString(key, "error"));
            else if (entry instanceof Integer)
                value = String.valueOf(sharedPreferences.getInt(key, 0));
        }

        Aware.setSetting(getApplicationContext(), key, value);
        Preference pref = findPreference(key);
        if (CheckBoxPreference.class.isInstance(pref)) {
            CheckBoxPreference check = (CheckBoxPreference) findPreference(key);
            check.setChecked(Aware.getSetting(getApplicationContext(), key).equals("true"));

            //update the parent to show active/inactive
            new Settings_Page.SettingsSync().execute(pref);

            //Start/Stop sensor
            Aware.startAWARE(getApplicationContext());
        }
        if (EditTextPreference.class.isInstance(pref)) {
            EditTextPreference text = (EditTextPreference) findPreference(key);
            text.setText(Aware.getSetting(getApplicationContext(), key));
        }
        if (ListPreference.class.isInstance(pref)) {
            ListPreference list = (ListPreference) findPreference(key);
            list.setSummary(list.getEntry());
        }
    }

    private void updatePluginIndicators() {
        // Retrieve all plugins in study config
        JSONObject studyConfig = Aware.getStudyConfig(getApplicationContext(), Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER));
        try {
            JSONArray pluginsList = studyConfig.getJSONArray("plugins");
            for (int i = 0; i < pluginsList.length(); i++) {
                JSONObject pluginConfig = pluginsList.getJSONObject(i);
                String pluginKey = pluginConfig.getString("plugin");
                //HACK: For most plugins so far
                String pluginName = pluginKey.substring(pluginKey.indexOf("plugin"));
                pluginName = pluginName.replaceAll("\\.", "_");
                Boolean pluginIsActive = Aware.getSetting(getApplicationContext(), "status_" + pluginName).equalsIgnoreCase("true");
                // Update plugin icons based on status
                Preference pluginPref = findPreference(pluginName);
                try {
                    Class res = R.drawable.class;
                    Field field = res.getField("ic_" + pluginName);
                    int icon_id = field.getInt(null);
                    Drawable category_icon = ContextCompat.getDrawable(getApplicationContext(), icon_id);
                    if (category_icon != null) {
                        int colorId = pluginIsActive ? R.color.settingEnabled : R.color.settingDisabled;
                        category_icon.setColorFilter(new PorterDuffColorFilter(ContextCompat.getColor(getApplicationContext(), colorId), PorterDuff.Mode.SRC_IN));
                        pluginPref.setIcon(category_icon);
                    }
                } catch (NoSuchFieldException | IllegalAccessException e) {
                    e.printStackTrace();
                }

            }
            // onContentChanged();
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        prefs.registerOnSharedPreferenceChangeListener(this);

        new Settings_Page.SettingsSync().executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR, //use all cores available to process UI faster
                findPreference(Aware_Preferences.DEVICE_ID),
                findPreference(Aware_Preferences.DEVICE_LABEL),
                findPreference(Aware_Preferences.AWARE_VERSION),
                findPreference(Aware_Preferences.STATUS_ACCELEROMETER),
                findPreference(Aware_Preferences.STATUS_APPLICATIONS),
                findPreference(Aware_Preferences.STATUS_BAROMETER),
                findPreference(Aware_Preferences.STATUS_BATTERY),
                findPreference(Aware_Preferences.STATUS_BLUETOOTH),
                findPreference(Aware_Preferences.STATUS_CALLS),
                findPreference(Aware_Preferences.STATUS_COMMUNICATION_EVENTS),
                findPreference(Aware_Preferences.STATUS_CRASHES),
                findPreference(Aware_Preferences.STATUS_ESM),
                findPreference(Aware_Preferences.STATUS_GRAVITY),
                findPreference(Aware_Preferences.STATUS_GYROSCOPE),
                findPreference(Aware_Preferences.STATUS_INSTALLATIONS),
                findPreference(Aware_Preferences.STATUS_KEYBOARD),
                findPreference(Aware_Preferences.STATUS_SCREENTEXT),
                findPreference(Aware_Preferences.STATUS_LIGHT),
                findPreference(Aware_Preferences.STATUS_LINEAR_ACCELEROMETER),
                findPreference(Aware_Preferences.STATUS_LOCATION_GPS),
                findPreference(Aware_Preferences.STATUS_LOCATION_NETWORK),
                findPreference(Aware_Preferences.STATUS_LOCATION_PASSIVE),
                findPreference(Aware_Preferences.STATUS_MAGNETOMETER),
                findPreference(Aware_Preferences.STATUS_MESSAGES),
                findPreference(Aware_Preferences.STATUS_MQTT),
                findPreference(Aware_Preferences.STATUS_NETWORK_EVENTS),
                findPreference(Aware_Preferences.STATUS_NETWORK_TRAFFIC),
                findPreference(Aware_Preferences.STATUS_NOTIFICATIONS),
                findPreference(Aware_Preferences.STATUS_PROCESSOR),
                findPreference(Aware_Preferences.STATUS_PROXIMITY),
                findPreference(Aware_Preferences.STATUS_ROTATION),
                findPreference(Aware_Preferences.STATUS_SCREEN),
                findPreference(Aware_Preferences.STATUS_SIGNIFICANT_MOTION),
                findPreference(Aware_Preferences.STATUS_TEMPERATURE),
                findPreference(Aware_Preferences.STATUS_TELEPHONY),
                findPreference(Aware_Preferences.STATUS_TIMEZONE),
                findPreference(Aware_Preferences.STATUS_WIFI),
                findPreference(Aware_Preferences.STATUS_WEBSERVICE),
                findPreference(Aware_Preferences.MQTT_SERVER),
                findPreference(Aware_Preferences.MQTT_PORT),
                findPreference(Aware_Preferences.MQTT_USERNAME),
                findPreference(Aware_Preferences.MQTT_PASSWORD),
                findPreference(Aware_Preferences.MQTT_KEEP_ALIVE),
                findPreference(Aware_Preferences.MQTT_QOS),
                findPreference(Aware_Preferences.WEBSERVICE_SERVER),
                findPreference(Aware_Preferences.FREQUENCY_WEBSERVICE),
                findPreference(Aware_Preferences.FREQUENCY_CLEAN_OLD_DATA),
                findPreference(Aware_Preferences.WEBSERVICE_CHARGING),
                findPreference(Aware_Preferences.WEBSERVICE_SILENT),
                findPreference(Aware_Preferences.WEBSERVICE_WIFI_ONLY),
                findPreference(Aware_Preferences.WEBSERVICE_FALLBACK_NETWORK),
                findPreference(Aware_Preferences.REMIND_TO_CHARGE),
                findPreference(Aware_Preferences.WEBSERVICE_SIMPLE),
                findPreference(Aware_Preferences.WEBSERVICE_REMOVE_DATA),
                findPreference(Aware_Preferences.DEBUG_DB_SLOW),
                findPreference(Aware_Preferences.FOREGROUND_PRIORITY),
                findPreference(Aware_Preferences.STATUS_TOUCH)
        );
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (prefs != null) prefs.unregisterOnSharedPreferenceChangeListener(this);
    }

    private class SettingsSync extends AsyncTask<Preference, Preference, Void> {
        @Override
        protected Void doInBackground(Preference... params) {
            for (Preference pref : params) {
                publishProgress(pref);
            }
            return null;
        }

        @Override
        protected void onProgressUpdate(Preference... values) {
            super.onProgressUpdate(values);

            Preference pref = values[0];

            if (pref != null) Log.i(TAG, "Syncing pref with key: " + pref.getKey());
            if (getPreferenceParent(pref) == null) return;

            if (CheckBoxPreference.class.isInstance(pref)) {
                CheckBoxPreference check = (CheckBoxPreference) findPreference(pref.getKey());
                check.setChecked(Aware.getSetting(getApplicationContext(), pref.getKey()).equals("true"));
                if (check.isChecked()) {
                    if (pref.getKey().equalsIgnoreCase(Aware_Preferences.STATUS_WEBSERVICE)) {
                        if (Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER).length() == 0) {
                            Toast.makeText(getApplicationContext(), "Study URL missing...", Toast.LENGTH_SHORT).show();
                        } else if (!Aware.isStudy(getApplicationContext())) {
                            //Shows UI to allow the user to join study
                            Intent joinStudy = new Intent(getApplicationContext(), Aware_Join_Study.class);
                            joinStudy.putExtra(Aware_Join_Study.EXTRA_STUDY_URL, Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER));
                            startActivity(joinStudy);
                        }
                    }
                    if (pref.getKey().equalsIgnoreCase(Aware_Preferences.FOREGROUND_PRIORITY)) {
                        sendBroadcast(new Intent(Aware.ACTION_AWARE_PRIORITY_FOREGROUND));
                    }
                } else {
                    if (pref.getKey().equalsIgnoreCase(Aware_Preferences.FOREGROUND_PRIORITY)) {
                        sendBroadcast(new Intent(Aware.ACTION_AWARE_PRIORITY_BACKGROUND));
                    }
                }
            }

            if (EditTextPreference.class.isInstance(pref)) {
                EditTextPreference text = (EditTextPreference) findPreference(pref.getKey());
                text.setText(Aware.getSetting(getApplicationContext(), pref.getKey()));
                text.setSummary(Aware.getSetting(getApplicationContext(), pref.getKey()));
            }

            if (ListPreference.class.isInstance(pref)) {
                ListPreference list = (ListPreference) findPreference(pref.getKey());
                list.setSummary(list.getEntry());
            }

            if (PreferenceScreen.class.isInstance(getPreferenceParent(pref))) {
                PreferenceScreen parent = (PreferenceScreen) getPreferenceParent(pref);

                boolean prefEnabled = Boolean.valueOf(Aware.getSetting(getApplicationContext(), Aware_Preferences.ENABLE_CONFIG_UPDATE));
                parent.setEnabled(prefEnabled);  // enabled/disabled based on config

                ListAdapter children = parent.getRootAdapter();
                boolean isActive = false;
                ArrayList sensorStatuses = new ArrayList<String>();
                for (int i = 0; i < children.getCount(); i++) {
                    Object obj = children.getItem(i);
                    if (CheckBoxPreference.class.isInstance(obj)) {
                        CheckBoxPreference child = (CheckBoxPreference) obj;
                        if (child.getKey().contains("status_")) {
                            sensorStatuses.add(child.getKey());
                            if (child.isChecked()) {
                                isActive = true;
                                break;
                            }
                        }
                    }
                }

                // Check if any of the status settings of a sensor (parent pref) is active in the study config
                JSONObject studyConfig = Aware.getStudyConfig(getApplicationContext(), Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER));
                boolean isActiveInConfig = false;
                try {
                    JSONArray sensorsList = studyConfig.getJSONArray("sensors");
                    for (int i = 0; i < sensorsList.length(); i++) {
                        JSONObject sensorInfo = sensorsList.getJSONObject(i);
                        String sensorSetting = sensorInfo.getString("setting");

                        if (sensorStatuses.contains(sensorSetting)) {
                            sensorStatuses.remove(sensorSetting);
                            isActiveInConfig = sensorInfo.getBoolean("value");
                        }

                        if (isActiveInConfig || sensorStatuses.size() == 0) break;
                    }
                } catch (JSONException e) {
                    e.printStackTrace();
                }

                // Only show sensor if it is active in the study config
                if (isActiveInConfig) {
                    if (pref != null) Log.i(TAG, "Pref with key: " + pref.getKey() + " is active!");
                    try {
                        Class res = R.drawable.class;
                        Field field = res.getField("ic_action_" + parent.getKey());
                        int icon_id = field.getInt(null);
                        Drawable category_icon = ContextCompat.getDrawable(getApplicationContext(), icon_id);
                        if (category_icon != null) {
                            int colorId = isActive ? R.color.settingEnabled : R.color.settingDisabled;
                            category_icon.setColorFilter(new PorterDuffColorFilter(ContextCompat.getColor(getApplicationContext(), colorId), PorterDuff.Mode.SRC_IN));
                            parent.setIcon(category_icon);
                            onContentChanged();
                        }
                    } catch (NoSuchFieldException | IllegalAccessException e) {
                        e.printStackTrace();
                    }
                } else {
                    PreferenceCategory rootSensorPref = (PreferenceCategory) getPreferenceParent(parent);
                    rootSensorPref.removePreference(parent);
                }
            }
            updatePluginIndicators();
        }
    }
}
