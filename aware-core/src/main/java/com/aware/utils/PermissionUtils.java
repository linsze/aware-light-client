package com.aware.utils;

import static com.aware.ui.PermissionsHandler.SERVICE_NAME;
import static com.aware.ui.PermissionsHandler.UNGRANTED_PERMISSIONS;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.preference.PreferenceActivity;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.R;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Utilities for managing permission requests and handling denied permissions.
 */
public class PermissionUtils {
    /**
     * Used to display rationale for required permissions.
     */
    public static HashMap<String, String[]> PERMISSION_DESCRIPTIONS = new HashMap<String, String[]>(){{
        put(Manifest.permission.GET_ACCOUNTS, new String[]{"Access to contacts", "Retrieve account used for data synchronization"});
        put(Manifest.permission.WRITE_EXTERNAL_STORAGE, new String[]{"Access to external storage", "Store data locally"});
        put(Manifest.permission.RECORD_AUDIO, new String[]{"Record audio", "Detect environmental audio"});
        put(Manifest.permission.ACCESS_FINE_LOCATION, new String[]{"Access to location", "Estimate approximate location"});
        put(Manifest.permission.ACCESS_COARSE_LOCATION, new String[]{"Access to location", "Estimate approximate location"});
        put(Manifest.permission.READ_PHONE_STATE, new String[]{"Access to phone state", "Detect ongoing calls (without call content) and cellular network"});
        put(Manifest.permission.READ_CALL_LOG, new String[]{"Access to call logs", "Track incoming/outgoing/missed calls without call content"});
        put(Manifest.permission.READ_SMS, new String[]{"Access to SMS messages", "Track incoming/outgoing messages without message content"});
    }};

    /**
     * Maps sensors to preferences for starting/stopping the service (currently only for those that require additional permissions)
     */
    private static final String classHeading = "com.aware.";
    public static HashMap<String, String> SENSOR_PREFERENCE_MAPPINGS = new HashMap<String, String>(){{
        put(classHeading + "Communication", Aware_Preferences.STATUS_COMMUNICATION_EVENTS);
        put(classHeading + "Bluetooth", Aware_Preferences.STATUS_BLUETOOTH);
        put(classHeading + "Locations", Aware_Preferences.STATUS_LOCATION_GPS);
        put(classHeading + "Telephony", Aware_Preferences.STATUS_TELEPHONY);
        put(classHeading + "WiFi", Aware_Preferences.STATUS_WIFI);

    }};

    /**
     * To send a broadcast for sensor preference changes when permissions are denied.
     */
    public static final String SENSOR_PREFERENCE_CHANGED = "SENSOR_PREFERENCE_CHANGED";
    public static final String SENSOR_PREFERENCE = "sensor_preference";


    /**
     * Checks if the current service is pending for permissions to be requested (is yet to be in the queue).
     * Adds the service into the queue if true.
     * @param serviceName name of the current service
     * @return true if the service is yet to be in the queue, false otherwise
     */
    public static boolean checkPermissionServiceQueue(Context context, String serviceName) {
        if (serviceName == "") {
            return false;
        }

        boolean isPending = false;
        String serviceQueueStr = Aware.getSetting(context, Aware_Preferences.PENDING_PERMISSION_SERVICE_QUEUE);
        ArrayList<String> serviceQueue = new ArrayList<>();
        if (!serviceQueueStr.equals("")) {
            serviceQueue = new ArrayList<>(Arrays.asList(serviceQueueStr.split(",")));
        }
        if (!serviceQueue.contains(serviceName)) {
            serviceQueue.add(serviceName);
            isPending = true;
            Aware.setSetting(context, Aware_Preferences.PENDING_PERMISSION_SERVICE_QUEUE, TextUtils.join(",", serviceQueue));
        }
        return isPending;
    }

    /**
     * Removes the current service from the queue after the necessary permissions have been requested.
     * @param serviceName name of the current service
     */
    public static void removeServiceFromPermissionQueue(Context context, String serviceName) {
        String serviceQueueStr = Aware.getSetting(context, Aware_Preferences.PENDING_PERMISSION_SERVICE_QUEUE);
        ArrayList<String> serviceQueue = new ArrayList<>();
        if (!serviceQueueStr.equals("")) {
            serviceQueue = new ArrayList<>(Arrays.asList(serviceQueueStr.split(",")));
        }
        if (serviceQueue.contains(serviceName)) {
            serviceQueue.remove(serviceName);
        }
        Aware.setSetting(context, Aware_Preferences.PENDING_PERMISSION_SERVICE_QUEUE, TextUtils.join(",", serviceQueue));
    }

    /**
     * Listens to broadcasts to display prompts for denied permissions.
     */
    public static class PermissionResultReceiver extends BroadcastReceiver {
        private PreferenceActivity currentActivity;

        public PermissionResultReceiver(PreferenceActivity activity) {
            this.currentActivity = activity;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            String service = intent.getStringExtra(SERVICE_NAME);
            ArrayList<String> pendingPermissions = intent.getStringArrayListExtra(UNGRANTED_PERMISSIONS);
            if (!pendingPermissions.isEmpty()) {
                //HACK: Scheduler is currently the only service with mandatory permissions (i.e. app cannot run without the permissions)
                if (service.contains("Scheduler")) {
                    new PermissionDialog(currentActivity, service, pendingPermissions, true).showDialog();
                } else {
                    new PermissionDialog(currentActivity, service, pendingPermissions, false).showDialog();
                }
            }
        }
    }

    public static class PermissionListAdapter extends RecyclerView.Adapter<PermissionListAdapter.ViewHolder> {
        private ArrayList<PermissionInfo> permissionList;

        public class ViewHolder extends RecyclerView.ViewHolder {
            public TextView tv_permission_name;
            public TextView tv_permission_description;

            public ViewHolder(View v) {
                super(v);
                tv_permission_name = (TextView) v.findViewById(R.id.permission_name);
                tv_permission_description = (TextView) v.findViewById(R.id.permission_description);
            }
        }

        public PermissionListAdapter(ArrayList<PermissionInfo> permissionList) {
            this.permissionList = permissionList;
        }

        @Override
        public PermissionListAdapter.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.permission_list_item, parent, false);

            ViewHolder vh = new ViewHolder(v);
            return vh;
        }

        @Override
        public void onBindViewHolder(ViewHolder holder, final int position) {
            String permissionName = permissionList.get(position).permissionName;
            holder.tv_permission_name.setText(permissionName);
            String description = permissionList.get(position).description;
            holder.tv_permission_description.setText(description);
        }

        @Override
        public int getItemCount() {
            return permissionList.size();
        }
    }

    public static class PermissionInfo {
        public String permissionName;
        public String description;

        public PermissionInfo(String permissionName, String description) {
            this.permissionName = permissionName;
            this.description = description;
        }
    }

    /**
     * Manages dialog to prompt reviewing of denied permissions.
     */
    public static class PermissionDialog extends DialogFragment {
        private Activity mActivity;
        private Context context;
        private String service;
        private ArrayList<String> pendingPermissions;
        private PermissionListAdapter permissionListAdapter;

        private Boolean isMandatory;


        public PermissionDialog(Activity activity, String service, ArrayList<String> pendingPermissions, Boolean isMandatory) {
            this.mActivity = activity;
            this.context = mActivity.getApplicationContext();
            this.service = service;
            this.pendingPermissions = pendingPermissions;
            this.isMandatory = isMandatory;
        }


        /**
         * Resets permission requests statuses to be prompted again.
         */
        private void resetPermissionStatuses() {
            String permissionRequestStatuses = Aware.getSetting(context, Aware_Preferences.PERMISSION_REQUEST_STATUSES);
            JSONObject permissionRequests = new JSONObject();
            try {
                if (!permissionRequestStatuses.equals("")) {
                    permissionRequests = new JSONObject(permissionRequestStatuses);
                }
                for (String p : pendingPermissions) {
                    permissionRequests.put(p, false);
                }
                Aware.setSetting(context, Aware_Preferences.PERMISSION_REQUEST_STATUSES, permissionRequests.toString());
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }

        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            AlertDialog.Builder builder = new AlertDialog.Builder(mActivity);
            LayoutInflater inflater = mActivity.getLayoutInflater();
            final View dialogView = inflater.inflate(R.layout.permission_dialog, null);

            // Application is unable to run if the permissions for the service are mandatory.
            String titleText;
            String description;
            if (isMandatory) {
                titleText = "Permissions needed";
                description = "The app is unable to run because the following permissions are denied.";
            } else {
                TextView warningText = dialogView.findViewById(R.id.exit_warning);
                warningText.setVisibility(View.GONE);
                if (service.contains("plugin")) {
                    service = service.substring(0, service.lastIndexOf("Plugin")-1);
                }
                String serviceName = service.substring(service.lastIndexOf('.')+1).replaceAll("_", " ");
                description = String.format("%s%s will be disabled because the following permissions are denied.", serviceName.substring(0, 1).toUpperCase(), serviceName.substring(1));
            }

            // Sets up list of denied permissions
            ArrayList<PermissionInfo> deniedPermissions = new ArrayList<>();
            boolean locationPermissionExists = false;
            for (String p: pendingPermissions) {
                //HACK: Prevent duplicated entries for coarse and fine location
                if (p.contains("LOCATION")) {
                    if (locationPermissionExists) {
                        continue;
                    } else {
                        locationPermissionExists = true;
                    }
                }
                String[] permissionDescriptions = PERMISSION_DESCRIPTIONS.get(p);
                if (permissionDescriptions != null) {
                    PermissionInfo permissionInfo = new PermissionInfo(permissionDescriptions[0], permissionDescriptions[1]);
                    deniedPermissions.add(permissionInfo);
                }
            }

            RecyclerView permissionsRecyclerView = (RecyclerView) dialogView.findViewById(R.id.service_permissions);
            RecyclerView.LayoutManager permissionsLayoutManager = new LinearLayoutManager(context);
            permissionsRecyclerView.setLayoutManager(permissionsLayoutManager);

            permissionListAdapter = new PermissionListAdapter(deniedPermissions);
            permissionsRecyclerView.setAdapter(permissionListAdapter);

            builder.setView(dialogView);
            builder.setTitle("Review Permissions")
                    .setMessage(description)
                    .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                        // Restart the relevant processes to prompt permission requests
                        public void onClick(DialogInterface dialog, int id) {
                            PermissionDialog.this.dismiss();
                            resetPermissionStatuses();
                            // This boolean is currently assumed to be false for sensors and plugins, and true for other background processes
                            if (isMandatory) {
                                //HACK: Other background processes are currently hardcoded
                                if (service.contains("Scheduler")) {
                                    Aware.startScheduler(context);
                                }
                            } else {
                                if (PluginsManager.isInstalled(context, service) != null) {
                                    Aware.startPlugin(context, service);
                                } else {
                                    String sensorPreference = SENSOR_PREFERENCE_MAPPINGS.get(service);
                                    if (sensorPreference != null) {
                                        Aware.startAWARE(context);
                                    }
                                }
                            }
                        }
                    })
                    .setNegativeButton("No", new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog, int id) {
                            PermissionDialog.this.dismiss();
                            // Exit the application if the permissions are mandatory for the service to run
                            if (isMandatory) {
                                mActivity.finishAffinity();
                                System.exit(0);
                            } else {
                            // Disable the sensor/plugin otherwise
                                if (PluginsManager.isInstalled(context, service) != null) {
                                    Aware.stopPlugin(context, service);
                                } else {
                                    String sensorPreference = SENSOR_PREFERENCE_MAPPINGS.get(service);
                                    if (sensorPreference != null) {
                                        Aware.setSetting(context, sensorPreference, false);
                                        Aware.startAWARE(context);
                                        // Broadcast changes to sensor preferences
                                        Intent sensorIntent = new Intent();
                                        sensorIntent.setAction(SENSOR_PREFERENCE_CHANGED);
                                        sensorIntent.putExtra(SENSOR_PREFERENCE, sensorPreference);
                                        mActivity.sendBroadcast(sensorIntent);
                                    }
                                }

                            }
                        }
                    });
            return builder.create();
        }

        public void showDialog() {
            this.setCancelable(false);
            this.show(mActivity.getFragmentManager(), "dialog");
        }

        @Override
        public void show(FragmentManager manager, String tag) {
            try {
                FragmentTransaction ft = manager.beginTransaction();
                ft.add(this, tag);
                ft.commitAllowingStateLoss();
            } catch (IllegalStateException e) {
                // Do nothing
            }
        }

        @Override
        public void onSaveInstanceState(Bundle outState) {
            //No call for super(). Bug on API Level > 11.
        }
    }
}
