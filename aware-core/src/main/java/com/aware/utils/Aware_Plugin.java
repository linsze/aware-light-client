
package com.aware.utils;

import static com.aware.ui.PermissionsHandler.ACTION_AWARE_PERMISSIONS_CHECK;
import static com.aware.ui.PermissionsHandler.PLUGIN_FULL_PERMISSIONS_NOT_GRANTED;
import static com.aware.ui.PermissionsHandler.SERVICE_NAME;
import static com.aware.ui.PermissionsHandler.UNGRANTED_PERMISSIONS;

import android.Manifest;
import android.app.Service;
import android.content.*;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;

import androidx.core.content.ContextCompat;
import androidx.core.content.PermissionChecker;
import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.ui.PermissionsHandler;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

/**
 * Aware_Plugin: Extend to integrate with the framework (extension of Android Service class).
 *
 * @author denzil
 */
public class Aware_Plugin extends Service {

    /**
     * Debug tag for this plugin
     */
    public String TAG = "AWARE Plugin";

    /**
     * Debug flag for this plugin
     */
    public boolean DEBUG = false;

    /**
     * Context producer for this plugin
     */
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
     * Plugin is inactive
     */
    public static final int STATUS_PLUGIN_OFF = 0;

    /**
     * Plugin is active
     */
    public static final int STATUS_PLUGIN_ON = 1;

    /**
     * Indicates if permissions were accepted OK
     */
    public boolean PERMISSIONS_OK = true;

    /**
     * Integration with sync adapters
     */
    public String AUTHORITY = "";

    @Override
    public void onCreate() {
        super.onCreate();

        //Register Context Broadcaster
        IntentFilter filter = new IntentFilter();
        filter.addAction(Aware.ACTION_AWARE_CURRENT_CONTEXT);
        filter.addAction(Aware.ACTION_AWARE_STOP_PLUGINS);
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
                Intent cantStartPluginIntent = new Intent();
                cantStartPluginIntent.setAction(PLUGIN_FULL_PERMISSIONS_NOT_GRANTED);
                cantStartPluginIntent.putExtra(SERVICE_NAME, getClass().getName());
                cantStartPluginIntent.putExtra(UNGRANTED_PERMISSIONS, PENDING_PERMISSIONS);
                sendBroadcast(cantStartPluginIntent);
                stopSelf();
                return START_NOT_STICKY;
            } else if (PENDING_PERMISSIONS.size() > 0) {
                Intent permissions = new Intent(this, PermissionsHandler.class);
                //HACK: Modified to only request for additional permissions that were not granted initially
                permissions.putExtra(PermissionsHandler.EXTRA_REQUIRED_PERMISSIONS, PENDING_PERMISSIONS);
//            permissions.putExtra(PermissionsHandler.EXTRA_REQUIRED_PERMISSIONS, REQUIRED_PERMISSIONS);
                permissions.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                permissions.putExtra(PermissionsHandler.EXTRA_REDIRECT_SERVICE, getApplicationContext().getPackageName() + "/" + getClass().getName()); //restarts plugin once permissions are accepted
                startActivity(permissions);
            }
        } else if (PERMISSIONS_OK) {
//            if (Aware.getSetting(this, Aware_Preferences.STATUS_WEBSERVICE).equals("true")) {
//                SSLManager.handleUrl(getApplicationContext(), Aware.getSetting(this, Aware_Preferences.WEBSERVICE_SERVER), true);
//            }
            //Restores core AWARE service in case it get's killed
            if (!Aware.IS_CORE_RUNNING) {
                Intent aware = new Intent(getApplicationContext(), Aware.class);
                startService(aware);
            }

            //Aware.startAWARE(getApplicationContext());

            //Aware.debug(this, "active: " + getClass().getName() + " package: " + getPackageName());
        }
        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        if (PERMISSIONS_OK) {
           //Aware.debug(this, "destroyed: " + getClass().getName() + " package: " + getPackageName());

           Aware.stopAWARE(getApplicationContext());
        }

        if (contextBroadcaster != null) unregisterReceiver(contextBroadcaster);
    }

    /**
     * Interface to share context with other applications/plugins<br/>
     * You are encouraged to broadcast your contexts here for reusability in other plugins and apps!
     *
     * @author denzil
     */
    public interface ContextProducer {
        void onContext();
    }

    /**
     * AWARE Context Broadcaster<br/>
     * - ACTION_AWARE_CURRENT_CONTEXT: returns current plugin's context
     * - ACTION_AWARE_STOP_PLUGINS: stops this plugin
     * - ACTION_AWARE_SYNC_DATA: sends the data to the server
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
            if (intent.getAction().equals(Aware.ACTION_AWARE_STOP_PLUGINS)) {
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

    private static ContextBroadcaster contextBroadcaster = null;

    @Override
    public IBinder onBind(Intent arg0) {
        return null;
    }
}
