package com.aware.phone.ui;

import static com.aware.Aware.TAG;
import static com.aware.utils.PermissionUtils.MULTIPLE_PREFERENCES_UPDATED;
import static com.aware.utils.PermissionUtils.SENSOR_PREFERENCE_MAPPINGS;
import static com.aware.utils.PermissionUtils.SERVICES_WITH_DENIED_PERMISSIONS;
import static com.aware.utils.PermissionUtils.SERVICE_FULL_PERMISSIONS_NOT_GRANTED;
import static com.aware.utils.PermissionUtils.SENSOR_PREFERENCE;
import static com.aware.utils.PermissionUtils.SENSOR_PREFERENCE_UPDATED;

import android.app.Dialog;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
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
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.phone.R;
import com.aware.utils.PermissionUtils;
import com.aware.utils.PluginsManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;

/**
 * Settings page to view and update study, sensors, and plugin configurations.
 */
public class Settings_Page extends Aware_Activity {

    private static SharedPreferences prefs;

    private PluginsListener pluginsListener;

    private SensorPreferenceListener sensorPreferenceListener = new SensorPreferenceListener();

    private final PermissionUtils.SingleServicePermissionReceiver singleServicePermissionReceiver = new PermissionUtils.SingleServicePermissionReceiver(Settings_Page.this);

    private final PermissionUtils.DeniedPermissionsReceiver deniedPermissionsReceiver = new PermissionUtils.DeniedPermissionsReceiver(Settings_Page.this);

    private ArrayList<String> currentPlugins = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("com.aware.phone", Context.MODE_PRIVATE);

        setContentView(R.layout.aware_light_main);
        // addPreferencesFromResource(R.xml.pref_aware_light);
        addPreferencesFromResource(R.xml.pref_sensors_plugins);
        setupPlugins();

        // Monitors for external changes in plugin's states and refresh the UI
        pluginsListener = new PluginsListener();

        IntentFilter filter = new IntentFilter();
        filter.addAction(Aware.ACTION_AWARE_UPDATE_PLUGINS_INFO);
        registerReceiver(pluginsListener, filter);

        IntentFilter permissionResults = new IntentFilter();
        permissionResults.addAction(SERVICE_FULL_PERMISSIONS_NOT_GRANTED);
        registerReceiver(singleServicePermissionReceiver, permissionResults);

        IntentFilter sensorPreferenceChanges = new IntentFilter();
        sensorPreferenceChanges.addAction(SENSOR_PREFERENCE_UPDATED);
        registerReceiver(sensorPreferenceListener, sensorPreferenceChanges);

        IntentFilter deniedPermissionsResults = new IntentFilter();
        deniedPermissionsResults.addAction(SERVICES_WITH_DENIED_PERMISSIONS);
        registerReceiver(deniedPermissionsReceiver, deniedPermissionsResults);

        Aware.setSetting(getApplicationContext(), Aware_Preferences.BULK_SERVICE_ACTIVATION, true);
    }

    /**
     * Listens to updates in plugin statuses to reflect color changes in UI.
     */
    private class PluginsListener extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(Aware.ACTION_AWARE_UPDATE_PLUGINS_INFO)) {
                updatePluginIndicators();
            }
        }
    }

    /**
     * Listens to changes in sensor preferences to reflect in UI of preference tree.
     */
    private class SensorPreferenceListener extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            String prefKey = intent.getStringExtra(SENSOR_PREFERENCE);
            Boolean multiplePref = intent.getBooleanExtra(MULTIPLE_PREFERENCES_UPDATED, false);
            if (multiplePref) {
                syncAllPreferences();
            } else {
                Preference sensorPref = findPreference(prefKey);
                if (sensorPref != null) {
                    new SettingsSync().execute(sensorPref);
                }
            }
        }
    }

    @Override
    public boolean onPreferenceTreeClick(PreferenceScreen preferenceScreen, final Preference preference) {
        if (preference instanceof PreferenceScreen) {
            Dialog subpref = ((PreferenceScreen) preference).getDialog();

            //HACK: Inflate plugin preferences into the current layout.
            if (subpref == null) {
                String prefKey = preference.getKey();
                String packageName = null;

                switch (prefKey) {
                    case "plugin_google_activity_recognition":
                        packageName = "com.aware.plugin.google.activity_recognition";
                        break;
                    case "plugin_ambient_noise":
                        packageName = "com.aware.plugin.ambient_noise";
                        break;
                    case "plugin_device_usage":
                        packageName = "com.aware.plugin.device_usage";
                        break;
                }
                if (packageName != null) {
                    String bundledPackage;
                    PackageInfo pkg = PluginsManager.isInstalled(getApplicationContext(), packageName);
                    if (pkg != null && pkg.versionName.equals("bundled")) {
                        bundledPackage = getApplicationContext().getPackageName();
                        Intent open_settings = new Intent();
                        open_settings.setComponent(new ComponentName(((bundledPackage.length() > 0) ? bundledPackage : packageName), packageName + ".Settings"));
                        startActivity(open_settings);
                    }
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

            // Update status of communication events if any of call or message status has been updated
            if (key.equals(Aware_Preferences.STATUS_CALLS) || key.equals(Aware_Preferences.STATUS_MESSAGES)) {
                boolean statusCalls = Aware.getSetting(getApplicationContext(), Aware_Preferences.STATUS_CALLS).equals("true");
                boolean statusMessages = Aware.getSetting(getApplicationContext(), Aware_Preferences.STATUS_MESSAGES).equals("true");
                Aware.setSetting(getApplicationContext(), Aware_Preferences.STATUS_COMMUNICATION_EVENTS, (statusCalls || statusMessages));
            } else if (key.equals(Aware_Preferences.STATUS_SCREEN)) {
                boolean screenStatus = Aware.getSetting(getApplicationContext(), Aware_Preferences.STATUS_SCREEN).equals("true");
                Aware.setSetting(getApplicationContext(), "status_plugin_device_usage", screenStatus);
                if (screenStatus) {
                    Aware.startPlugin(getApplicationContext(), "com.aware.plugin.device_usage");
                } else {
                    Aware.stopPlugin(getApplicationContext(), "com.aware.plugin.device_usage");
                }
            }

            //Start/Stop sensor
//            Aware.startAWARE(getApplicationContext());
            //NOTE: Only start/stop the relevant sensor
            Aware.activateSensorFromPreference(getApplicationContext(), key);

            //update the parent to show active/inactive
            new Settings_Page.SettingsSync().execute(pref);
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

    /**
     * Retrieves and stores all plugins listed in the study configuration (regardless of whether they are enabled or not).
     * Starts the plugin services if they are enabled.
     */
    private void setupPlugins() {
        JSONObject studyConfig = Aware.getStudyConfig(getApplicationContext(), Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER));
        try {
            JSONArray pluginsList = studyConfig.getJSONArray("plugins");
            for (int i = 0; i < pluginsList.length(); i++) {
                JSONObject pluginConfig = pluginsList.getJSONObject(i);
                String pluginKey = pluginConfig.getString("plugin");
                currentPlugins.add(pluginKey);

                //HACK: For most plugins so far
                String pluginName = pluginKey.substring(pluginKey.indexOf("plugin"));
                pluginName = pluginName.replaceAll("\\.", "_");
                Boolean pluginIsEnabled = Aware.getSetting(getApplicationContext(), "status_" + pluginName).equalsIgnoreCase("true");
                if (pluginIsEnabled) {
                    Aware.startPlugin(getApplicationContext(), pluginKey);
                }
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }

        // HACK: Hide device_usage preference selection and set it to be enabled/disabled based on screen events
        Preference deviceUsagePref = findPreference("plugin_device_usage");
        PreferenceCategory rootPref = (PreferenceCategory) getPreferenceParent(deviceUsagePref);
        rootPref.removePreference(deviceUsagePref);
    }

    /**
     * Update color indicators corresponding to whether plugins are enabled.
     */
    private void updatePluginIndicators() {
        for (String plugin: currentPlugins) {
            String pluginName = plugin.substring(plugin.indexOf("plugin"));
            pluginName = pluginName.replaceAll("\\.", "_");
            Boolean pluginIsActive = Aware.getSetting(getApplicationContext(), "status_" + pluginName).equalsIgnoreCase("true");
            // Update plugin icons based on status
            Preference pluginPref = findPreference(pluginName);
            if (pluginPref != null) {
                try {
                    Class res = R.drawable.class;
                    Field field = res.getField("ic_" + pluginName);
                    int icon_id = field.getInt(null);
                    Drawable category_icon = ContextCompat.getDrawable(getApplicationContext(), icon_id);
                    if (category_icon != null) {
                        int colorId = pluginIsActive ? R.color.settingEnabled : R.color.settingDisabled;
                        category_icon.setColorFilter(new PorterDuffColorFilter(ContextCompat.getColor(getApplicationContext(), colorId), PorterDuff.Mode.SRC_IN));
                        pluginPref.setIcon(category_icon);
                        onContentChanged();
                    }
                } catch (NoSuchFieldException | IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void syncAllPreferences() {
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
    protected void onResume() {
        super.onResume();
        prefs.registerOnSharedPreferenceChangeListener(this);

        syncAllPreferences();
        updatePluginIndicators();

        // Check if there are denied permissions during the first bulk sensor activation
        Boolean isFirstBulkActivation = (Aware.getSetting(getApplicationContext(), Aware_Preferences.BULK_SERVICE_ACTIVATION)).equals("true");
        if (isFirstBulkActivation) {
            String deniedPermissionsStr = Aware.getSetting(getApplicationContext(), Aware_Preferences.DENIED_PERMISSIONS_SERVICES);
            if (!deniedPermissionsStr.equals("")) {
                Intent deniedPermissionsIntent = new Intent();
                deniedPermissionsIntent.setAction(SERVICES_WITH_DENIED_PERMISSIONS);
                sendBroadcast(deniedPermissionsIntent);
            }
        }

        // Rechecks permissions after resuming from application settings from 2 different scenarios
        // 1 - Resuming after updating one or more permissions from a bulk activation (multiple preferences involved)
        if (Aware.getSetting(getApplicationContext(), Aware_Preferences.REDIRECTED_TO_LOCAL_PERMISSIONS).equals("true")) {
            // Reset the state
            Aware.setSetting(getApplicationContext(), Aware_Preferences.REDIRECTED_TO_LOCAL_PERMISSIONS, false);
            try {
                String curDeniedPermissionsStr = Aware.getSetting(getApplicationContext(), Aware_Preferences.DENIED_PERMISSIONS_SERVICES);
                JSONObject deniedPermissions = new JSONObject(curDeniedPermissionsStr);
                Iterator<String> permissionIterator = deniedPermissions.keys();

                while (permissionIterator.hasNext()) {
                    String permission = permissionIterator.next();
                    // Check for newly granted permissions and remove from the global state
                    if (ActivityCompat.checkSelfPermission(getApplicationContext(), permission) == PackageManager.PERMISSION_GRANTED) {
                        permissionIterator.remove();
                    }
                }

                // Update global state based on the latest checks
                Aware.setSetting(getApplicationContext(), Aware_Preferences.DENIED_PERMISSIONS_SERVICES, deniedPermissions.toString());
                PermissionUtils.disableDeniedPermissionPreferences(getApplicationContext(), this);
                updatePluginIndicators();
                syncAllPreferences();
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }
        // 2 - Resuming after updating permissions for a single preference
        else if (Aware.getSetting(getApplicationContext(), Aware_Preferences.REDIRECTED_TO_LOCAL_PERMISSIONS_FROM_SINGLE_PREFERENCE).equals("true")) {
            Aware.setSetting(getApplicationContext(), Aware_Preferences.REDIRECTED_TO_LOCAL_PERMISSIONS_FROM_SINGLE_PREFERENCE, false);
            String servicePref = Aware.getSetting(getApplicationContext(), Aware_Preferences.REDIRECTED_SERVICE);
            String servicePermissionStr = Aware.getSetting(getApplicationContext(), Aware_Preferences.REDIRECTED_PERMISSIONS);
            ArrayList<String> servicePermissions = new ArrayList<>();
            if (!servicePermissionStr.equals("")) {
                servicePermissions = new ArrayList<>(Arrays.asList(servicePermissionStr.split(",")));
            }

            JSONObject deniedPermissions = null;
            try {
                String curDeniedPermissionsStr = Aware.getSetting(getApplicationContext(), Aware_Preferences.DENIED_PERMISSIONS_SERVICES);
                deniedPermissions = new JSONObject(curDeniedPermissionsStr);
            } catch (JSONException e) {
                e.printStackTrace();
            }

            boolean allPermissionsGranted = true;
            for (String p: servicePermissions) {
                if (!(ActivityCompat.checkSelfPermission(getApplicationContext(), p) == PackageManager.PERMISSION_GRANTED)) {
                    allPermissionsGranted = false;
                    break;
                } else if (deniedPermissions != null && deniedPermissions.has(p))  {
                    deniedPermissions.remove(p);
                }
            }

            if (deniedPermissions != null) {
                Aware.setSetting(getApplicationContext(), Aware_Preferences.DENIED_PERMISSIONS_SERVICES, deniedPermissions.toString());
            }

            if (!allPermissionsGranted) {
                PermissionUtils.disableService(getApplicationContext(), servicePref, this);
                updatePluginIndicators();
                syncAllPreferences();
            } else {
                if (PluginsManager.isInstalled(getApplicationContext(), servicePref) != null) {
                    Aware.startPlugin(getApplicationContext(), servicePref);
                    updatePluginIndicators();
                } else {
                    for (ArrayList<String> sensorPrefs: SENSOR_PREFERENCE_MAPPINGS.values()) {
                        if (sensorPrefs.contains(servicePref)) {
                            Aware.activateSensorFromPreference(getApplicationContext(), servicePref);
                            new SettingsSync().execute(findPreference(servicePref));
                            break;
                        }
                    }
                }
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (prefs != null) prefs.unregisterOnSharedPreferenceChangeListener(this);
    }

    /**
     * Initializes state and value of a specific preference.
     */
    private void initializePreferenceValue(Preference preference) {
        if (CheckBoxPreference.class.isInstance(preference)) {
            CheckBoxPreference check = (CheckBoxPreference) findPreference(preference.getKey());
            check.setChecked(Aware.getSetting(getApplicationContext(), preference.getKey()).equals("true"));
        }

        if (EditTextPreference.class.isInstance(preference)) {
            EditTextPreference text = (EditTextPreference) findPreference(preference.getKey());
            if (text != null) {
                text.setText(Aware.getSetting(getApplicationContext(), preference.getKey()));
                text.setSummary(Aware.getSetting(getApplicationContext(), preference.getKey()));
            }
        }

        if (ListPreference.class.isInstance(preference)) {
            String prefString = preference.getKey();
            ListPreference list = (ListPreference) findPreference(prefString);
            String freq = Aware.getSetting(getApplicationContext(), prefString);
            list.setValue(freq);
            int entryIndex = list.findIndexOfValue(freq);
            CharSequence entryString = list.getEntries()[entryIndex];
            list.setSummary(entryString);
        }
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterReceiver(pluginsListener);
        unregisterReceiver(singleServicePermissionReceiver);
        unregisterReceiver(sensorPreferenceListener);
        unregisterReceiver(deniedPermissionsReceiver);
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

            initializePreferenceValue(pref);
            if (CheckBoxPreference.class.isInstance(pref)) {
                CheckBoxPreference check = (CheckBoxPreference) findPreference(pref.getKey());
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

            if (PreferenceScreen.class.isInstance(getPreferenceParent(pref))) {
                PreferenceScreen parent = (PreferenceScreen) getPreferenceParent(pref);

                boolean prefEnabled = Boolean.valueOf(Aware.getSetting(getApplicationContext(), Aware_Preferences.ENABLE_CONFIG_UPDATE));
                parent.setEnabled(prefEnabled);  // enabled/disabled based on config

                ListAdapter children = parent.getRootAdapter();
                boolean isActive = false;
                ArrayList sensorStatuses = new ArrayList<String>();
                for (int i = 0; i < children.getCount(); i++) {
                    Object obj = children.getItem(i);
                    initializePreferenceValue((Preference) obj);
                    if (CheckBoxPreference.class.isInstance(obj)) {
                        CheckBoxPreference child = (CheckBoxPreference) obj;
                        if (child.getKey().contains("status_")) {
                            sensorStatuses.add(child.getKey());
                            if (child.isChecked()) {
                                isActive = true;
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
//                            isActiveInConfig = sensorInfo.getBoolean("value");
                            // HACK: Updated to retain sensor preference even though they might be disabled by default
                            isActiveInConfig = true;
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
        }
    }
}
