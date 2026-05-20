package com.aware.phone.ui;

import android.Manifest;
import android.app.AppOpsManager;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.preference.Preference;
import android.preference.PreferenceGroup;
import android.preference.PreferenceManager;
import android.preference.PreferenceScreen;
import android.provider.Settings;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.ListView;

import com.aware.Applications;
import com.aware.ApplicationUsage;
import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.phone.R;
import com.aware.ui.PermissionsHandler;
import com.aware.utils.PermissionUtils;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;
import androidx.core.content.PermissionChecker;

import static com.aware.Aware.AWARE_NOTIFICATION_IMPORTANCE_GENERAL;
import static com.aware.Aware.TAG;
import static com.aware.Aware.setNotificationProperties;
import static com.aware.Aware.startApplicationUsage;
import static com.aware.Aware.stopApplicationUsage;
import static com.aware.utils.PermissionUtils.SERVICES_WITH_DENIED_PERMISSIONS;
import static com.aware.utils.PermissionUtils.SERVICE_FULL_PERMISSIONS_NOT_GRANTED;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Main page (Home) that provides the study description and navigation instructions.
 */
public class Aware_Light_Client extends Aware_Activity {
    private static SharedPreferences prefs;
    public static boolean permissions_ok;

    private static Hashtable<Integer, Boolean> listSensorType;

    private static final ArrayList<String> REQUIRED_PERMISSIONS = new ArrayList<>();
    private static final Hashtable<String, Integer> optionalSensors = new Hashtable<>();

    private final Aware.AndroidPackageMonitor packageMonitor = new Aware.AndroidPackageMonitor();

    private final PermissionUtils.SingleServicePermissionReceiver singleServicePermissionReceiver = new PermissionUtils.SingleServicePermissionReceiver(Aware_Light_Client.this);
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("com.aware.phone", Context.MODE_PRIVATE);

        // Initialize views
//        setContentView(R.layout.activity_aware_light);
        setContentView(R.layout.aware_light_main);
        addPreferencesFromResource(R.xml.pref_home);
        // Remove listview item separator and adjust top margin specifically for the current layout
        ListView overallListView = getListView();
        overallListView.setDivider(null);
        ViewGroup.MarginLayoutParams layoutParams = (ViewGroup.MarginLayoutParams) overallListView
                .getLayoutParams();
        layoutParams.setMargins(layoutParams.leftMargin, 0, layoutParams.rightMargin, layoutParams.bottomMargin);
        startAwareService();
//        hideUnusedPreferences();
    }

    private void startAwareService() {
        // Initialize and check optional sensors and required permissions before starting AWARE service
        optionalSensors.put(Aware_Preferences.STATUS_ACCELEROMETER, Sensor.TYPE_ACCELEROMETER);
        optionalSensors.put(Aware_Preferences.STATUS_SIGNIFICANT_MOTION, Sensor.TYPE_ACCELEROMETER);
        optionalSensors.put(Aware_Preferences.STATUS_BAROMETER, Sensor.TYPE_PRESSURE);
        optionalSensors.put(Aware_Preferences.STATUS_GRAVITY, Sensor.TYPE_GRAVITY);
        optionalSensors.put(Aware_Preferences.STATUS_GYROSCOPE, Sensor.TYPE_GYROSCOPE);
        optionalSensors.put(Aware_Preferences.STATUS_LIGHT, Sensor.TYPE_LIGHT);
        optionalSensors.put(Aware_Preferences.STATUS_LINEAR_ACCELEROMETER, Sensor.TYPE_LINEAR_ACCELERATION);
        optionalSensors.put(Aware_Preferences.STATUS_MAGNETOMETER, Sensor.TYPE_MAGNETIC_FIELD);
        optionalSensors.put(Aware_Preferences.STATUS_PROXIMITY, Sensor.TYPE_PROXIMITY);
        optionalSensors.put(Aware_Preferences.STATUS_ROTATION, Sensor.TYPE_ROTATION_VECTOR);
        optionalSensors.put(Aware_Preferences.STATUS_TEMPERATURE, Sensor.TYPE_AMBIENT_TEMPERATURE);

        SensorManager manager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        List<Sensor> sensors = manager.getSensorList(Sensor.TYPE_ALL);
        listSensorType = new Hashtable<>();
        for (int i = 0; i < sensors.size(); i++) {
            listSensorType.put(sensors.get(i).getType(), true);
        }

        //NOTE: Only request for currently necessary permissions at the beginning
        REQUIRED_PERMISSIONS.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        REQUIRED_PERMISSIONS.add(Manifest.permission.ACCESS_WIFI_STATE);
        REQUIRED_PERMISSIONS.add(Manifest.permission.GET_ACCOUNTS);
        REQUIRED_PERMISSIONS.add(Manifest.permission.WRITE_SYNC_SETTINGS);
        REQUIRED_PERMISSIONS.add(Manifest.permission.READ_SYNC_SETTINGS);
        REQUIRED_PERMISSIONS.add(Manifest.permission.READ_SYNC_STATS);
        REQUIRED_PERMISSIONS.add(Manifest.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);

        //NOTE: Commented out permissions are only required for specific sensors
//        REQUIRED_PERMISSIONS.add(Manifest.permission.CAMERA);
//        REQUIRED_PERMISSIONS.add(Manifest.permission.BLUETOOTH);
//        REQUIRED_PERMISSIONS.add(Manifest.permission.BLUETOOTH_ADMIN);
//        REQUIRED_PERMISSIONS.add(Manifest.permission.ACCESS_COARSE_LOCATION);
//        REQUIRED_PERMISSIONS.add(Manifest.permission.ACCESS_FINE_LOCATION);
//        REQUIRED_PERMISSIONS.add(Manifest.permission.READ_PHONE_STATE);


        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) REQUIRED_PERMISSIONS.add(Manifest.permission.FOREGROUND_SERVICE);

        boolean PERMISSIONS_OK = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            for (String p : REQUIRED_PERMISSIONS) {
                if (ActivityCompat.checkSelfPermission(this, p) != PackageManager.PERMISSION_GRANTED) {
                    PERMISSIONS_OK = false;
                    break;
                }
            }
        }
        if (PERMISSIONS_OK) {
            Intent aware = new Intent(this, Aware.class);
            startService(aware);
        }

        IntentFilter awarePackages = new IntentFilter();
        awarePackages.addAction(Intent.ACTION_PACKAGE_ADDED);
        awarePackages.addAction(Intent.ACTION_PACKAGE_REMOVED);
        awarePackages.addDataScheme("package");
        registerReceiver(packageMonitor, awarePackages);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Intent whitelisting = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
            whitelisting.setData(Uri.parse("package:" + getPackageName()));
            startActivity(whitelisting);
        }

        IntentFilter permissionResults = new IntentFilter();
        permissionResults.addAction(SERVICE_FULL_PERMISSIONS_NOT_GRANTED);
        registerReceiver(singleServicePermissionReceiver, permissionResults);
    }

    @Override
    protected void onResume() {
        super.onResume();

        permissions_ok = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            for (String p : REQUIRED_PERMISSIONS) {
                if (PermissionChecker.checkSelfPermission(this, p) != PermissionChecker.PERMISSION_GRANTED) {
                    permissions_ok = false;
                    break;
                }
            }
        }

        if (!permissions_ok) {
            Log.d(TAG, "Requesting permissions...");

            Intent permissionsHandler = new Intent(this, PermissionsHandler.class);
            permissionsHandler.putStringArrayListExtra(PermissionsHandler.EXTRA_REQUIRED_PERMISSIONS, REQUIRED_PERMISSIONS);
            permissionsHandler.putExtra(PermissionsHandler.EXTRA_REDIRECT_ACTIVITY, getPackageName() + "/" + getClass().getName());
            permissionsHandler.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(permissionsHandler);

        } else {
            if (prefs.getAll().isEmpty() && Aware.getSetting(getApplicationContext(), Aware_Preferences.DEVICE_ID).length() == 0) {
                PreferenceManager.setDefaultValues(getApplicationContext(), "com.aware.phone", Context.MODE_PRIVATE, R.xml.aware_preferences, true);
                prefs.edit().commit();
            } else {
                PreferenceManager.setDefaultValues(getApplicationContext(), "com.aware.phone", Context.MODE_PRIVATE, R.xml.aware_preferences, false);
            }

            Map<String, ?> defaults = prefs.getAll();
            for (Map.Entry<String, ?> entry : defaults.entrySet()) {
                if (Aware.getSetting(getApplicationContext(), entry.getKey(), "com.aware.phone").length() == 0) {
                    Aware.setSetting(getApplicationContext(), entry.getKey(), entry.getValue(), "com.aware.phone"); //default AWARE settings
                }
            }

            if (Aware.getSetting(getApplicationContext(), Aware_Preferences.DEVICE_ID).length() == 0) {
                UUID uuid = UUID.randomUUID();
                Aware.setSetting(getApplicationContext(), Aware_Preferences.DEVICE_ID, uuid.toString(), "com.aware.phone");
            }

            if (Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER).length() == 0) {
                Aware.setSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER, "http://api.awareframework.com/index.php");
            }

            Set<String> keys = optionalSensors.keySet();
            for (String optionalSensor : keys) {
                Preference pref = findPreference(optionalSensor);
                PreferenceGroup parent = getPreferenceParent(pref);
                if (pref != null && parent != null && pref.getKey().equalsIgnoreCase(optionalSensor) && !listSensorType.containsKey(optionalSensors.get(optionalSensor)))
                    parent.setEnabled(false);
            }

            try {
                PackageInfo awareInfo = getApplicationContext().getPackageManager().getPackageInfo(getApplicationContext().getPackageName(), PackageManager.GET_ACTIVITIES);
                Aware.setSetting(getApplicationContext(), Aware_Preferences.AWARE_VERSION, awareInfo.versionName);
            } catch (PackageManager.NameNotFoundException e) {
                e.printStackTrace();
            }

            //Check if AWARE is active on the accessibility services. Android Wear doesn't support accessibility services (no API yet...)
//            if (!Aware.is_watch(this)) {
//                Applications.isAccessibilityServiceActive(this);
//            }

            //Check if AWARE is allowed to run on Doze
            //Aware.isBatteryOptimizationIgnored(this, getPackageName());
        }

        /**
         * Checks if usage access has been allowed
         */
        if (Aware.getSetting(getApplicationContext(), Aware_Preferences.STATUS_APPLICATION_USAGE).equals("true")) {
            AppOpsManager appOps = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
            int mode = appOps.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, android.os.Process.myUid(), getPackageName());
            boolean granted = (mode == AppOpsManager.MODE_ALLOWED);
            if (!granted) {
                // Stop the service first if it was previous started but usage access has been removed
                if (Aware.isApplicationUsageActive()) {
                    stopApplicationUsage(getApplicationContext());
                }
                // Notification to prompt permission
                ApplicationUsage.promptUsageAccessNotification(getApplicationContext());
            } else if (!Aware.isApplicationUsageActive()) {
                // Cases where access has just been allowed
                startApplicationUsage(getApplicationContext());
            }
        }
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String s) {
        // Do nothing
    }

    public static boolean isBatteryOptimizationIgnored(Context context, String package_name) {
        boolean is_ignored = true;
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP_MR1) {
            PowerManager pm = (PowerManager) context.getApplicationContext().getSystemService(Context.POWER_SERVICE);
            is_ignored = pm.isIgnoringBatteryOptimizations(package_name);
        }

        if (!is_ignored) {
            NotificationCompat.Builder mBuilder = new NotificationCompat.Builder(context, Aware.AWARE_NOTIFICATION_CHANNEL_GENERAL);
            mBuilder.setSmallIcon(com.aware.R.drawable.ic_stat_aware_recharge);
            mBuilder.setContentTitle(context.getApplicationContext().getResources().getString(com.aware.R.string.aware_activate_battery_optimize_ignore_title));
            mBuilder.setContentText(context.getApplicationContext().getResources().getString(com.aware.R.string.aware_activate_battery_optimize_ignore));
            mBuilder.setAutoCancel(true);
            mBuilder.setOnlyAlertOnce(true); //notify the user only once
            mBuilder.setDefaults(NotificationCompat.DEFAULT_ALL);
            mBuilder = setNotificationProperties(mBuilder, AWARE_NOTIFICATION_IMPORTANCE_GENERAL);

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                mBuilder.setChannelId(Aware.AWARE_NOTIFICATION_CHANNEL_GENERAL);

            Intent batteryIntent = new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS);
            batteryIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);

            PendingIntent clickIntent = PendingIntent.getActivity(context, 0, batteryIntent, PendingIntent.FLAG_UPDATE_CURRENT);
            mBuilder.setContentIntent(clickIntent);

            NotificationManager notManager = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
            notManager.notify(Aware.AWARE_BATTERY_OPTIMIZATION_ID, mBuilder.build());
        }

        Log.d(Aware.TAG, "Battery Optimizations: " + is_ignored);

        return is_ignored;
    }

    @Override
    protected void onStop() {
        // Check if the activity is finishing
        boolean isFinishing = this.isFinishing();

        // Handle based on whether it's system-initiated closure
        if (!isFinishing) {
            if (isBatteryOptimizationIgnored(this, "com.aware.phone")) {
                Log.d("AWARE-Light_Client", "AWARE-Light stopped from background: may be caused by battery optimization");
                Aware.debug(this, "AWARE-Light stopped from background: may be caused by battery optimization");
            } else {
                Log.d("AWARE-Light_Client", "AWARE-Light stopped from background: may be caused by system settings");
                Aware.debug(this, "AWARE-Light stopped from background: may be caused by system settings");
            }
        }
        super.onStop();
    }


    @Override
    protected void onDestroy() {
        // Check if the activity is finishing
        boolean isFinishing = this.isFinishing();

        // Handle based on whether it's user-initiated or system-initiated closure
        if (isFinishing) {
            // User initiated closure
            Aware.debug(this, "AWARE-Light interface cleaned from the list of frequently used apps");
        }
        Log.d("AWARE-Light_Client", "AWARE-Light interface cleaned from the list of frequently used apps");
        super.onDestroy();
        unregisterReceiver(packageMonitor);
        unregisterReceiver(singleServicePermissionReceiver);
    }

    private void hideUnusedPreferences() {
        Preference dataExchangePref = findPreference("data_exchange");
        if (dataExchangePref != null) {
            PreferenceScreen rootSensorPref = (PreferenceScreen) getPreferenceParent(dataExchangePref);
            rootSensorPref.removePreference(dataExchangePref);
        }
    }
}
