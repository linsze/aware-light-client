package com.aware.ui;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import androidx.core.app.ActivityCompat;
import com.aware.Aware;
import com.aware.Aware_Preferences;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

/**
 * This is an invisible activity used to request the needed permissions from the user from API 23 onwards.
 * Created by denzil on 22/10/15.
 */
public class PermissionsHandler extends Activity {

    private String TAG = "PermissionsHandler";

    /**
     * Extra ArrayList<String> with Manifest.permission that require explicit users' permission on Android API 23+
     */
    public static final String EXTRA_REQUIRED_PERMISSIONS = "required_permissions";

    /**
     * Class name of the Activity redirect, e.g., Class.getClass().getName();
     */
    public static final String EXTRA_REDIRECT_ACTIVITY = "redirect_activity";

    /**
     * Class name of the Service redirect, e.g., Class.getClass().getName();
     */
    public static final String EXTRA_REDIRECT_SERVICE = "redirect_service";

    /**
     * Used on redirect service to know when permissions have been accepted
     */
    public static final String ACTION_AWARE_PERMISSIONS_CHECK = "ACTION_AWARE_PERMISSIONS_CHECK";

    /**
     * Used to trigger a dialog that the plugin may be disabled due to denied permissions
     */
    public static final String PLUGIN_FULL_PERMISSIONS_NOT_GRANTED = "PLUGIN_FULL_PERMISSIONS_NOT_GRANTED";

    /**
     * Used to trigger a dialog that the application may stop due to denied mandatory permissions
     */
    public static final String SCHEDULER_FULL_PERMISSIONS_NOT_GRANTED = "SCHEDULER_FULL_PERMISSIONS_NOT_GRANTED";

    public static final String SERVICE_NAME = "service_name";
    public static String UNGRANTED_PERMISSIONS = "ungranted_permissions";

    /**
     * The request code for the permissions
     */
    public static final int RC_PERMISSIONS = 112;

    private Intent redirect_activity, redirect_service;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d("Permissions", "Permissions request for " + getPackageName());
    }

    @Override
    protected void onResume() {
        super.onResume();
        Intent currentIntent = getIntent();

        if (currentIntent != null && currentIntent.getExtras() != null && currentIntent.getSerializableExtra(EXTRA_REQUIRED_PERMISSIONS) != null) {
            ArrayList<String> permissionsNeeded = (ArrayList<String>) currentIntent.getSerializableExtra(EXTRA_REQUIRED_PERMISSIONS);
            ArrayList<String> pendingPermissions = new ArrayList<>();

            // Check for permissions that are yet to be requested
            JSONObject allStatuses = new JSONObject();
            String permissionsRequestStatus = Aware.getSetting(getApplicationContext(), Aware_Preferences.PERMISSION_REQUEST_STATUSES);
            try {
                if (!permissionsRequestStatus.equals("")) {
                    allStatuses = new JSONObject(permissionsRequestStatus);
                }
                for (String p : permissionsNeeded) {
                    if (!allStatuses.has(p) || !allStatuses.getBoolean(p)) {
                        pendingPermissions.add(p);
                    }
                }
            } catch (JSONException e) {
                // Do nothing
            }

            //HACK: Finish the current requests if all pending permissions have been requested before
            if (pendingPermissions.isEmpty()) {
                finish();
            } else {
                ActivityCompat.requestPermissions(PermissionsHandler.this, pendingPermissions.toArray(new String[pendingPermissions.size()]), RC_PERMISSIONS);
                try {
                    for (String p: pendingPermissions) {
                        allStatuses.put(p, true);
                    }
                    Aware.setSetting(getApplicationContext(), Aware_Preferences.PERMISSION_REQUEST_STATUSES, allStatuses.toString());
                } catch (JSONException e) {
                    throw new RuntimeException(e);
                }
                if (currentIntent.hasExtra(EXTRA_REDIRECT_ACTIVITY)) {
                    redirect_activity = new Intent();
                    String[] component = currentIntent.getStringExtra(EXTRA_REDIRECT_ACTIVITY).split("/");
                    redirect_activity.setComponent(new ComponentName(component[0], component[1]));
                    redirect_activity.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                } else if (currentIntent.hasExtra(EXTRA_REDIRECT_SERVICE)) {
                    redirect_service = new Intent();
                    redirect_service.setAction(ACTION_AWARE_PERMISSIONS_CHECK);
                    String[] component = currentIntent.getStringExtra(EXTRA_REDIRECT_SERVICE).split("/");
                    redirect_service.setComponent(new ComponentName(component[0], component[1]));
                }
            }
        } else {
            Intent activity = new Intent();
            setResult(Activity.RESULT_OK, activity);
            finish();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        ArrayList<String> ungrantedPermissions = new ArrayList<>();
        if (requestCode == RC_PERMISSIONS) {
            int not_granted = 0;
            for (int i = 0; i < permissions.length; i++) {
                if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                    ungrantedPermissions.add(permissions[i]);
                    not_granted++;
                    Log.d(Aware.TAG, permissions[i] + " was not granted");
                } else {
                    Log.d(Aware.TAG, permissions[i] + " was granted");
                }
            }

            if (not_granted > 0) {
                if (redirect_activity == null) {
                    Intent activity = new Intent();
                    setResult(Activity.RESULT_CANCELED, activity);
                }
                if (redirect_activity != null) {
                    setResult(Activity.RESULT_CANCELED, redirect_activity);
                    redirect_activity.putExtra(SERVICE_NAME, redirect_activity.getClass().getName());
                    redirect_activity.putExtra(UNGRANTED_PERMISSIONS, ungrantedPermissions);
                    //HACK: Prevent activity and service from starting multiple times
//                    startActivity(redirect_activity);
                }
//                if (redirect_service != null) {
//                    startService(redirect_service);
//                }
            } else {
                if (redirect_activity == null) {
                    Intent activity = new Intent();
                    setResult(Activity.RESULT_OK, activity);
                } else if (redirect_activity != null) {
                    setResult(Activity.RESULT_OK, redirect_activity);
                }
            }
            finish();
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (redirect_service != null) {
            Log.d(TAG, "Redirecting to Service: " + redirect_service.getComponent().toString());
//            redirect_service.setAction(ACTION_AWARE_PERMISSIONS_CHECK);
            startService(redirect_service);
        }
        if (redirect_activity != null) {
            Log.d(TAG, "Redirecting to Activity: " + redirect_activity.getComponent().toString());
//            setResult(Activity.RESULT_OK, redirect_activity);
            startActivity(redirect_activity);
        }
        Log.d("Permissions", "Handled permissions for " + getPackageName());
    }
}
