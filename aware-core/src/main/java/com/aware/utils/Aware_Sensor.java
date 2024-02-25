package com.aware.utils;

import static androidx.core.app.ActivityCompat.startActivityForResult;
import static com.aware.ui.PermissionsHandler.ACTION_AWARE_PERMISSIONS_CHECK;
import static com.aware.ui.PermissionsHandler.RC_PERMISSIONS;
import static com.aware.ui.PermissionsHandler.SERVICE_FULL_PERMISSIONS_NOT_GRANTED;
import static com.aware.ui.PermissionsHandler.SERVICE_NAME;
import static com.aware.ui.PermissionsHandler.UNGRANTED_PERMISSIONS;

import android.Manifest;
import android.app.Service;
import android.content.*;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.text.TextUtils;
import android.util.Log;

import androidx.core.content.ContextCompat;
import androidx.core.content.PermissionChecker;
import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.ui.PermissionsHandler;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Aware_Sensor: Extend to integrate with the framework (extension of Android Service class).
 *
 * @author dferreira
 */
public class Aware_Sensor extends Service {

    /**
     * Debug tag for this sensor
     */
    public String TAG = "AWARE Sensor";

    /**
     * Debug flag for this sensor
     */
    public boolean DEBUG = false;

    public ContextProducer CONTEXT_PRODUCER = null;

    /**
     * Permissions needed for this plugin to run
     */
    public ArrayList<String> REQUIRED_PERMISSIONS = new ArrayList<>();

    /**
     * Permissions currently yet to be granted
     */
    private ArrayList<String> PENDING_PERMISSIONS = new ArrayList<>();

    /**
     * Indicates if permissions were accepted OK
     */
    public boolean PERMISSIONS_OK = true;


    /**
     * Integration with sync adapters
     */
    public String AUTHORITY = "";

    /**
     * Interface to share context with other applications/addons<br/>
     * You MUST broadcast your contexts here!
     *
     * @author denzil
     */
    public interface ContextProducer {
        void onContext();
    }

    @Override
    public void onCreate() {
        super.onCreate();

        //Register Context Broadcaster
        IntentFilter filter = new IntentFilter();
        filter.addAction(Aware.ACTION_AWARE_CURRENT_CONTEXT);
        filter.addAction(Aware.ACTION_AWARE_STOP_SENSORS);
        filter.addAction(Aware.ACTION_AWARE_SYNC_DATA);

        if (contextBroadcaster == null) {
            contextBroadcaster = new ContextBroadcaster(CONTEXT_PRODUCER, TAG, AUTHORITY);
        }

        registerReceiver(contextBroadcaster, filter);

        REQUIRED_PERMISSIONS.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        REQUIRED_PERMISSIONS.add(Manifest.permission.GET_ACCOUNTS);
        REQUIRED_PERMISSIONS.add(Manifest.permission.WRITE_SYNC_SETTINGS);
        REQUIRED_PERMISSIONS.add(Manifest.permission.READ_SYNC_SETTINGS);
        REQUIRED_PERMISSIONS.add(Manifest.permission.READ_SYNC_STATS);

        Log.d(Aware.TAG, "created: " + getClass().getName() + " package: " + getPackageName());
    }

    /**
     * Checks if all required permissions have been granted and keeps track if they have been requested before.
     */
    private void checkPermissionRequests() {
        PERMISSIONS_OK = true;
        PENDING_PERMISSIONS = new ArrayList<>();

        //HACK: Store into a list of pending permissions if they have not been requested before
        String permissionRequestStatuses = Aware.getSetting(getApplicationContext(), Aware_Preferences.PERMISSION_REQUEST_STATUSES);
        JSONObject permissionRequests = new JSONObject();
        try {
            if (!permissionRequestStatuses.equals("")) {
                permissionRequests = new JSONObject(permissionRequestStatuses);
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                for (String p : REQUIRED_PERMISSIONS) {
                    //HACK: https://stackoverflow.com/questions/44813943/should-i-prefer-contextcompat-or-permissionchecker-for-permission-checking-on-an
                    // if (PermissionChecker.checkSelfPermission(this, p) != PermissionChecker.PERMISSION_GRANTED) {
                    if (ContextCompat.checkSelfPermission(this, p) != PackageManager.PERMISSION_GRANTED) {
                        PERMISSIONS_OK = false;
                        PENDING_PERMISSIONS.add(p);
                        // Stores status if permission has been requested before or by other services
                        if (!permissionRequests.has(p) || !permissionRequests.getBoolean(p)) {
                            permissionRequests.put(p, false);
                        }
                    }
                }
            }
            Aware.setSetting(getApplicationContext(), Aware_Preferences.PERMISSION_REQUEST_STATUSES, permissionRequests.toString());
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        checkPermissionRequests();

        //HACK: Not all permissions are granted but they have been requested
        // Send broadcast to activity to display a dialog
        if (!PERMISSIONS_OK) {
            if (intent != null && intent.getAction() != null && intent.getAction().equals(ACTION_AWARE_PERMISSIONS_CHECK)) {
                Intent cantRunSchedulerIntent = new Intent(SERVICE_FULL_PERMISSIONS_NOT_GRANTED);
                cantRunSchedulerIntent.putExtra(SERVICE_NAME, getClass().getName());
                cantRunSchedulerIntent.putExtra(UNGRANTED_PERMISSIONS, PENDING_PERMISSIONS);
                sendBroadcast(cantRunSchedulerIntent);
                stopSelf();
                return START_NOT_STICKY;
            } else if (PENDING_PERMISSIONS.size() > 0) {
                if (PermissionUtils.checkPermissionServiceQueue(getApplicationContext(), getClass().getName())) {
                    Intent permissions = new Intent(this, PermissionsHandler.class);
                    //HACK: Modified to only request for additional permissions that were not granted initially
                    permissions.putExtra(PermissionsHandler.EXTRA_REQUIRED_PERMISSIONS, PENDING_PERMISSIONS);
//            permissions.putExtra(PermissionsHandler.EXTRA_REQUIRED_PERMISSIONS, REQUIRED_PERMISSIONS);
                    permissions.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    permissions.putExtra(PermissionsHandler.EXTRA_REDIRECT_SERVICE, getApplicationContext().getPackageName() + "/" + getClass().getName()); //restarts plugin once permissions are accepted
                    startActivity(permissions);
                } else {
                    stopSelf();
                    return START_NOT_STICKY;
                }
            }
        } else {
//            if (Aware.getSetting(this, Aware_Preferences.STATUS_WEBSERVICE).equals("true") && Aware.getSetting(this, Aware_Preferences.WEBSERVICE_SERVER).contains("https")) {
//                downloadCertificate(this);
//            }
            //Aware.debug(this, "active: " + getClass().getName() + " package: " + getPackageName());
        }

        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (PERMISSIONS_OK) {
            //Aware.debug(this, "destroyed: " + getClass().getName() + " package: " + getPackageName());
        }

        //Unregister Context Broadcaster
        if (contextBroadcaster != null) unregisterReceiver(contextBroadcaster);
    }

    /**
     * Attempts to parse integer preference entry, resets to default value if the entry is invalid.
     * @param prefString key of preference entry
     * @param defaultValue default preference value for resetting
     */
    public void tryParseIntPreference(String prefString, int defaultValue) {
        try {
            String freq = Aware.getSetting(getApplicationContext(), prefString);
            int freqInt = Integer.parseInt(freq);
            Aware.setSetting(getApplicationContext(), prefString, freqInt);
        } catch (NumberFormatException e) {
            Aware.setSetting(getApplicationContext(), prefString, defaultValue);
        }
    }

    /**
     * Attempts to parse double preference entry, resets to default value if the entry is invalid.
     * @param prefString key of preference entry
     * @param defaultValue default preference value for resetting
     */
    public void tryParseDoublePreference(String prefString, Double defaultValue) {
        try {
            String freq = Aware.getSetting(getApplicationContext(), prefString);
            Double freqDouble = Double.parseDouble(freq);
            Aware.setSetting(getApplicationContext(), prefString, freqDouble);
        } catch (NumberFormatException e) {
            Aware.setSetting(getApplicationContext(), prefString, defaultValue);
        }
    }

    /**
     * AWARE Context Broadcaster<br/>
     * - ACTION_AWARE_CURRENT_CONTEXT: returns current plugin's context
     * - ACTION_AWARE_SYNC_DATA: push content provider data remotely
     * - ACTION_AWARE_CLEAR_DATA: clears local and remote database
     * - ACTION_AWARE_STOP_SENSORS: stops this sensor
     * - ACTION_AWARE_SPACE_MAINTENANCE: clears old data from content providers
     *
     * @author denzil
     */
    public static class ContextBroadcaster extends BroadcastReceiver {

        private ContextProducer cp;
        private String tag;
        private String provider;

        public ContextBroadcaster(ContextProducer contextProducer, String logcatTag, String providerAuthority) {
            this.cp = contextProducer;
            this.tag = logcatTag;
            this.provider = providerAuthority;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(Aware.ACTION_AWARE_CURRENT_CONTEXT)) {
                if (cp != null) {
                    cp.onContext();
                }
            }
            if (intent.getAction().equals(Aware.ACTION_AWARE_STOP_SENSORS)) {
                if (Aware.DEBUG) Log.d(tag, tag + " stopped");
                try {
                    Intent self = new Intent(context, Class.forName(context.getApplicationContext().getClass().getName()));
                    context.stopService(self);
                } catch (ClassNotFoundException e) {
                    e.printStackTrace();
                }
            }
            if (intent.getAction().equals(Aware.ACTION_AWARE_SYNC_DATA) && provider.length() > 0) {
                Bundle sync = new Bundle();
                sync.putBoolean(ContentResolver.SYNC_EXTRAS_MANUAL, true);
                sync.putBoolean(ContentResolver.SYNC_EXTRAS_EXPEDITED, true);
                ContentResolver.requestSync(Aware.getAWAREAccount(context), provider, sync);
            }
        }
    }

//    private void downloadCertificate(Context context) {
//        new SSLDownloadTask().execute(context);
//    }

//    class SSLDownloadTask extends AsyncTask<Context, Void, Void>
//    {
//        @Override
//        protected Void doInBackground(Context... params) {
//            SSLManager.handleUrl(getApplicationContext(), Aware.getSetting(params[0], Aware_Preferences.WEBSERVICE_SERVER), true);
//            return null;
//        }
//    }

    private static ContextBroadcaster contextBroadcaster = null;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
