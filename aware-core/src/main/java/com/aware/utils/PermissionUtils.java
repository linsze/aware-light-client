package com.aware.utils;

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
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.preference.PreferenceActivity;
import android.provider.Settings;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.TextUtils;
import android.text.style.BulletSpan;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.R;

import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

/**
 * Utilities for managing permission requests and handling denied permissions.
 * @author Lin Sze Khoo
 */
public class PermissionUtils {
    /**
     * Reference for static string declarations:
     * 1. Intent/broadcast action strings are fully capitalized
     * 2. The remaining (lower case) strings are for extras to pass additional data in intents.
     */

    /**
     * Used to broadcast that all mandatory permissions have been granted (and to only start the main intent after that).
     */
    public static final String MANDATORY_PERMISSIONS_GRANTED = "MANDATORY_PERMISSIONS_GRANTED";

    /**
     * Used to trigger a dialog that necessary permissions for a specific service are not fully granted
     */
    public static final String SERVICE_FULL_PERMISSIONS_NOT_GRANTED = "SERVICE_FULL_PERMISSIONS_NOT_GRANTED";

    public static final String SERVICE_NAME = "service_name";

    public static final String UNGRANTED_PERMISSIONS = "ungranted_permissions";

    /**
     * Used to trigger a dialog that one or more services will be affected due to specific denied permission
     */
    public static final String SERVICES_WITH_DENIED_PERMISSIONS = "SERVICES_WITH_DENIED_PERMISSIONS";

    /**
     * To send a broadcast for plugin changes to be reflected in UI when permissions are denied.
     * NOTE: Currently only listened to by ambient noise plugin since it is the only one requiring additional permission.
     */
    public static final String PLUGIN_PREFERENCE_UPDATED = "PLUGIN_PREFERENCE_UPDATED";
    public static final String UPDATED_PLUGIN = "updated_plugin";

    /**
     * To send a broadcast for sensor preference changes to be reflected in UI when permissions are denied.
     */
    public static final String SENSOR_PREFERENCE_UPDATED = "SENSOR_PREFERENCE_UPDATED";
    public static final String SENSOR_PREFERENCE = "sensor_preference";

    public static final String MULTIPLE_PREFERENCES_UPDATED = "multiple_preference_updated";

    /**
     * Used to display rationale for required permissions.
     */
    public static HashMap<String, String[]> PERMISSION_DESCRIPTIONS = new HashMap<String, String[]>(){{
        put(Manifest.permission.GET_ACCOUNTS, new String[]{"Contacts", "database sync"});
        put(Manifest.permission.WRITE_EXTERNAL_STORAGE, new String[]{"Files", "local data storage"});
        put(Manifest.permission.RECORD_AUDIO, new String[]{"Microphone", "audio detection"});
        put(Manifest.permission.ACCESS_FINE_LOCATION, new String[]{"Precise location", "more precise location estimate"});
        put(Manifest.permission.ACCESS_COARSE_LOCATION, new String[]{"Rough location", "approximate location estimate"});
        put(Manifest.permission.READ_PHONE_STATE, new String[]{"Phone", "call detection"});
        put(Manifest.permission.READ_CALL_LOG, new String[]{"Call logs", "call logging"});
        put(Manifest.permission.READ_SMS, new String[]{"SMS", "SMS logging"});
    }};

    /**
     * Maps sensors to preferences for starting/stopping the service (currently only for those that require additional permissions)
     */
    private static final String classHeading = "com.aware.";
    public static HashMap<String, ArrayList<String>> SENSOR_PREFERENCE_MAPPINGS = new HashMap<String, ArrayList<String>>(){{
        put(classHeading + "Communication", new ArrayList<String>(){{add(Aware_Preferences.STATUS_CALLS); add(Aware_Preferences.STATUS_MESSAGES);}});
        put(classHeading + "Bluetooth", new ArrayList<String>(){{add(Aware_Preferences.STATUS_BLUETOOTH);}});
        put(classHeading + "Locations", new ArrayList<String>(){{add(Aware_Preferences.STATUS_LOCATION_GPS); add(Aware_Preferences.STATUS_LOCATION_NETWORK); add(Aware_Preferences.STATUS_LOCATION_PASSIVE);}});
    }};

    /**
     * Checks if the current service is pending for permissions to be requested (is yet to be in the queue).
     * Adds the service into the queue if true.
     * @param serviceName name of the current service (sensor or plugin class)
     * @return true if the service is yet to be in the queue, false otherwise
     */
    public static boolean checkPermissionServiceQueue(Context context, String serviceName) {
        if (serviceName.equals("")) {
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
     * Removes service corresponding to the input preference from the queue after the necessary permissions have been requested.
     * @param pref current preference of a plugin or sensor
     * @param isClassName boolean of whether the input belongs to the class name of a sensor or plugin
     */
    public static String removeServiceFromPermissionQueue(Context context, String pref, Boolean isClassName) {
        String serviceQueueStr = Aware.getSetting(context, Aware_Preferences.PENDING_PERMISSION_SERVICE_QUEUE);
        ArrayList<String> serviceQueue = new ArrayList<>();
        if (!serviceQueueStr.equals("")) {
            serviceQueue = new ArrayList<>(Arrays.asList(serviceQueueStr.split(",")));
        }

        String prefClass = null;
        if (pref.contains("Scheduler") && serviceQueue.contains(pref)) {
            serviceQueue.remove(pref);
        } else {
            // Retrieve sensor or plugin corresponding to the preference
            if (PluginsManager.isInstalled(context, pref) != null) {
                prefClass = pref;
            } else {
                if (isClassName && SENSOR_PREFERENCE_MAPPINGS.keySet().contains(pref)) {
                    prefClass = pref;
                } else {
                    for (String sensor: SENSOR_PREFERENCE_MAPPINGS.keySet()) {
                        ArrayList<String> sensorPrefs = SENSOR_PREFERENCE_MAPPINGS.get(sensor);
                        if (sensorPrefs.contains(pref)) {
                            prefClass = sensor;
                            break;
                        }
                    }
                }
            }

            if (prefClass != null && serviceQueue.contains(prefClass)) {
                serviceQueue.remove(prefClass);
            }
        }
        Aware.setSetting(context, Aware_Preferences.PENDING_PERMISSION_SERVICE_QUEUE, TextUtils.join(",", serviceQueue));
        return prefClass;
    }

    /**
     * Adds a specific sensor class package name with the corresponding denied permissions to be displayed as prompt.
     * @param context application context
     * @param serviceName Class name corresponding to a sensor or plugin
     * @param pendingPermissions List of required permissions that have been denied
     */
    public static void addServiceToDeniedPermission(Context context, String serviceName, ArrayList<String> pendingPermissions) {
        String deniedPermissionsStr = Aware.getSetting(context, Aware_Preferences.DENIED_PERMISSIONS_SERVICES);
        JSONObject deniedPermissions = new JSONObject();
        try {
            if (!deniedPermissionsStr.equals("")) {
                deniedPermissions = new JSONObject(deniedPermissionsStr);
            }
            for (String p : pendingPermissions) {
                ArrayList<String> services = new ArrayList<>();
                if (deniedPermissions.has(p)) {
                    services = new ArrayList<>(Arrays.asList(deniedPermissions.getString(p).split(",")));
                }

                // NOTE: If service is a plugin, add it directly
                if (PluginsManager.isInstalled(context, serviceName) != null) {
                    if (!services.contains(serviceName)) {
                        services.add(serviceName);
                    }
                } else {
                    // NOTE: If service is a sensor class, retrieve and add individual preferences that are affected by the current permission.
                    try {
                        // Access class by name to get settings affected by the specific permission
                        Class<?> sensorClass = Class.forName(serviceName);
                        Object classInstance = sensorClass.newInstance();
                        // HACK: Field name is currently hardcoded
                        Field settingsPermissionsField = classInstance.getClass().getField("SETTINGS_PERMISSIONS");
                        HashMap<String, HashMap<String, String>> settingsPermissionsMap = (HashMap<String, HashMap<String, String>>) settingsPermissionsField.get(classInstance);
                        if (settingsPermissionsMap.get(p) != null) {
                            HashMap<String, String> permissionSettings = settingsPermissionsMap.get(p);
                            for (String settingPref: permissionSettings.values()) {
                                if (!services.contains(settingPref) && Aware.getSetting(context, settingPref).equals("true")) {
                                    services.add(settingPref);
                                }
                            }
                        }
                    } catch (ClassNotFoundException | ClassCastException | NoSuchFieldException |
                             IllegalAccessException | java.lang.InstantiationException e) {
                        e.printStackTrace();
                    }
                }
                deniedPermissions.put(p, TextUtils.join(",", services));
            }
            Aware.setSetting(context, Aware_Preferences.DENIED_PERMISSIONS_SERVICES, deniedPermissions.toString());
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    /**
     * Resets permission requests statuses to be prompted again.
     */
    public static void resetPermissionStatuses(Context context, ArrayList<String> pendingPermissions) {
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

    /**
     * Start an activity to open local application settings.
     */
    private static void redirectToLocalSettings(String preferenceString, Context context, Activity activity) {
        Aware.setSetting(context, preferenceString, true);
        Intent redirectToSettings = new Intent();
        redirectToSettings.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        Uri uri = Uri.fromParts("package", activity.getPackageName(), null);
        redirectToSettings.setData(uri);
        activity.startActivity(redirectToSettings);
    }

    public static Toast permissionToast(String toastMessage, Context context) {
        Toast permissionToast = Toast.makeText(context, toastMessage, Toast.LENGTH_LONG);
        View view = permissionToast.getView();
        TextView text = (TextView) view.findViewById(android.R.id.message);
        text.setTextColor(Color.RED);
        return permissionToast;
    }

    /**
     * Disables plugin or sensor by updating global preferences and sending broadcasts to reflect necessary changes.
     * NOTE: Service could be a preference for either a plugin or sensor.
     */
    public static void disableService(Context context, String servicePref, Activity activity) {
        if (servicePref.contains("plugin")) {
            String pluginName = "com.aware." + servicePref.substring(servicePref.indexOf("plugin")).replaceFirst("_", "\\.");
            if (PluginsManager.isInstalled(context, pluginName) != null) {
//            String pluginPref = service.substring(service.indexOf("plugin")).replaceAll("\\.", "_");
//            Aware.setSetting(context, "status_" + pluginPref, false);
                Aware.setSetting(context, servicePref, false);
                Aware.stopPlugin(context, pluginName);

                // NOTE: Sending broadcast is necessary to reflect the changes in checkboxes
                // Assuming that this function is triggered when failing to activate plugin manually
                Intent pluginIntent = new Intent();
                pluginIntent.setAction(PLUGIN_PREFERENCE_UPDATED);
                pluginIntent.putExtra(UPDATED_PLUGIN, pluginName);
                activity.sendBroadcast(pluginIntent);
            }
        } else {
            for (ArrayList<String> sensorPrefs: SENSOR_PREFERENCE_MAPPINGS.values()) {
                if (sensorPrefs.contains(servicePref)) {
                    Aware.setSetting(context, servicePref, false);
                    Aware.setSetting(context, Aware_Preferences.BULK_SERVICE_ACTIVATION, false);
                    Aware.activateSensorFromPreference(context, servicePref);

                    // Broadcast changes to sensor preferences
                    Intent sensorIntent = new Intent();
                    sensorIntent.setAction(SENSOR_PREFERENCE_UPDATED);
                    sensorIntent.putExtra(MULTIPLE_PREFERENCES_UPDATED, false);
                    sensorIntent.putExtra(SENSOR_PREFERENCE, servicePref);
                    activity.sendBroadcast(sensorIntent);

                    break;
                }
            }
        }
    }

    /**
     * Disable preferences affected by all currently denied permissions.
     * Remove denied permissions from the global state to prevent them from being prompted again.
     */
    public static void disableDeniedPermissionPreferences(Context context, Activity activity) {
        try {
            String curDeniedPermissionsStr = Aware.getSetting(context, Aware_Preferences.DENIED_PERMISSIONS_SERVICES);
            JSONObject deniedPermissions = new JSONObject(curDeniedPermissionsStr);
            Iterator<String> permissionIterator = deniedPermissions.keys();

            while (permissionIterator.hasNext()) {
                String permission = permissionIterator.next();
                // Disable sensing if their permissions are still not updated
                ArrayList<String> preferences = new ArrayList<>(Arrays.asList(deniedPermissions.getString(permission).split(",")));
                for (String pref: preferences) {
                    boolean isSensorPref = false;
                    // Check if it is a sensor preference
                    for (ArrayList<String> sensorPrefs: SENSOR_PREFERENCE_MAPPINGS.values()) {
                        if (sensorPrefs.contains(pref)) {
                            Aware.setSetting(context, pref, false);
                            isSensorPref = true;
                            break;
                        }
                    }
                    if (!isSensorPref) {
                        try {
                            if (PluginsManager.isInstalled(context, pref) != null) {
                                String pluginPref = "status_" + pref.substring(pref.indexOf("plugin")).replaceAll("\\.", "_");
                                Aware.setSetting(context, pluginPref, false);
                                Aware.stopPlugin(context, pref);
                            }
                        } catch (StringIndexOutOfBoundsException e) {
                            // Do nothing since it's not a valid plugin name
                        }
                    }
                }
            }
            // Broadcast changes to sensor preferences
            Intent sensorIntent = new Intent();
            sensorIntent.setAction(SENSOR_PREFERENCE_UPDATED);
            sensorIntent.putExtra(MULTIPLE_PREFERENCES_UPDATED, true);
            activity.sendBroadcast(sensorIntent);
            Aware.startAWARE(context, false);

            // Resets global denied permissions
            Aware.setSetting(context, Aware_Preferences.DENIED_PERMISSIONS_SERVICES, "");

        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    /**
     * Listens to broadcasts to display prompts when necessary permissions for a specific service have been denied.
     * NOTE: Used for scheduler or when a specific sensor/plugin is enabled manually.
     */
    public static class SingleServicePermissionReceiver extends BroadcastReceiver {
        private PreferenceActivity currentActivity;

        public SingleServicePermissionReceiver(PreferenceActivity activity) {
            this.currentActivity = activity;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            String service = intent.getStringExtra(SERVICE_NAME);
            ArrayList<String> pendingPermissions = intent.getStringArrayListExtra(UNGRANTED_PERMISSIONS);
            //HACK: Scheduler is currently the only service with mandatory permissions (i.e. app cannot run without the permissions)
            if (service.contains("Scheduler") || service.contains("com.aware.phone.ui.")) {
                new ReviewServiceDialog(currentActivity, service, pendingPermissions, true).showDialog();
            } else {
                //NOTE: Map service to the corresponding preference first
                String servicePref = null;
                String serviceClassName = service;
                if (PluginsManager.isInstalled(context, service) != null) {
                    serviceClassName = serviceClassName + ".Plugin";
                }

                // Access class by name to get preference descriptions
                try {
                    Class<?> serviceClass = Class.forName(serviceClassName);
                    Object classInstance = serviceClass.newInstance();
                    // HACK: Field name is currently hardcoded
                    Field settingsPermissionsField = classInstance.getClass().getField("SETTINGS_PERMISSIONS");
                    HashMap<String, HashMap<String, String>> settingsPermissionsMap = (HashMap<String, HashMap<String, String>>) settingsPermissionsField.get(classInstance);
                    ArrayList<String> candidatePrefs = new ArrayList<>();
                    for (String permission: pendingPermissions) {
                        ArrayList<String> allPermissionPrefs = new ArrayList<>(settingsPermissionsMap.get(permission).values());
                        if (candidatePrefs.size() == 0) {
                            candidatePrefs = allPermissionPrefs;
                        } else {
                            for (String pref: candidatePrefs) {
                                if (!allPermissionPrefs.contains(pref)) {
                                    candidatePrefs.remove(pref);
                                }
                            }
                        }
                    }
                    if (candidatePrefs.size() > 0) {
                        // Remove service from the queue
                        servicePref = candidatePrefs.get(0);
                    }
                } catch (ClassNotFoundException | ClassCastException | NoSuchFieldException |
                         IllegalAccessException | java.lang.InstantiationException e) {
                    e.printStackTrace();
                }

                if (servicePref != null) {
                    new ReviewServiceDialog(currentActivity, servicePref, pendingPermissions, false).showDialog();
                }
            }
        }
    }

    /**
     * Manages dialog to prompt reviewing of denied permissions for a specific service.
     */
    public static class ReviewServiceDialog extends DialogFragment {
        private Activity mActivity;
        private Context context;

        private String servicePreference;
        private ArrayList<String> pendingPermissions;
        private PermissionListAdapter permissionListAdapter;

        private Boolean isMandatory;


        public ReviewServiceDialog(Activity activity, String servicePreference, ArrayList<String> pendingPermissions, Boolean isMandatory) {
            this.mActivity = activity;
            this.context = mActivity.getApplicationContext();
            this.servicePreference = servicePreference;
            this.pendingPermissions = pendingPermissions;
            this.isMandatory = isMandatory;
        }

        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            AlertDialog.Builder builder = new AlertDialog.Builder(mActivity);
            LayoutInflater inflater = mActivity.getLayoutInflater();
            final View dialogView = inflater.inflate(R.layout.review_service_dialog, null);

            // Application is unable to run if the permissions for the service are mandatory.
            String titleText;
            String description;
            if (isMandatory) {
                description = "The app is unable to run because the following permissions are denied.";
            } else {
                TextView warningText = dialogView.findViewById(R.id.exit_warning);
                warningText.setText("Selecting NO will disable the sensing on your behalf.");
//                if (service.contains("plugin")) {
//                    service = service.substring(0, service.lastIndexOf("Plugin")-1);
//                }
//                String serviceName = service.substring(service.lastIndexOf('.')+1).replaceAll("_", " ");
//                description = String.format("%s%s sensing will be disabled because the following permissions are denied.", serviceName.substring(0, 1).toUpperCase(), serviceName.substring(1));
                description = "Selected sensing is unable to start because the following permissions are denied.";
            }

            // Sets up list of denied permissions
            ArrayList<ServicePermissionInfo> deniedPermissions = new ArrayList<>();
            for (String p: pendingPermissions) {
                String[] permissionDescriptions = PermissionUtils.PERMISSION_DESCRIPTIONS.get(p);
                if (permissionDescriptions != null) {
                    ServicePermissionInfo servicePermissionInfo = new ServicePermissionInfo(permissionDescriptions[0], "For " + permissionDescriptions[1]);
                    deniedPermissions.add(servicePermissionInfo);
                }
            }

            RecyclerView permissionsRecyclerView = (RecyclerView) dialogView.findViewById(R.id.service_permissions);
            RecyclerView.LayoutManager permissionsLayoutManager = new LinearLayoutManager(context);
            permissionsRecyclerView.setLayoutManager(permissionsLayoutManager);

            permissionListAdapter = new PermissionListAdapter(deniedPermissions);
            permissionsRecyclerView.setAdapter(permissionListAdapter);

            // Remove service from the queue
            PermissionUtils.removeServiceFromPermissionQueue(context, servicePreference, false);

            // Exit the application if the permissions are mandatory for the service to run
            builder.setView(dialogView);
            builder.setTitle("Review Permissions")
                    .setMessage(description)
                    .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                        // Restart the relevant processes to prompt permission requests
                        public void onClick(DialogInterface dialog, int id) {
                            ReviewServiceDialog.this.dismiss();
                            resetPermissionStatuses(context, pendingPermissions);
                            if (isMandatory) {
                                //HACK: Other background processes are currently hardcoded
                                if (servicePreference.contains("Scheduler")) {
                                    Aware.startScheduler(context);
                                }
                            } else {
                                Aware.setSetting(context, Aware_Preferences.REDIRECTED_SERVICE, servicePreference);
                                Aware.setSetting(context, Aware_Preferences.REDIRECTED_PERMISSIONS, TextUtils.join(",", pendingPermissions));
                                redirectToLocalSettings(Aware_Preferences.REDIRECTED_TO_LOCAL_PERMISSIONS_FROM_SINGLE_PREFERENCE, context, mActivity);
                            }
                        }
                    })
                    .setNegativeButton("No", new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog, int id) {
                            ReviewServiceDialog.this.dismiss();
                            if (isMandatory) {
                                // Exit the application
                                mActivity.finishAffinity();
                                System.exit(0);
                            } else {
                                disableService(context, servicePreference, mActivity);
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


    /**
     * Listens to broadcasts to display prompts when one or more services are affected due to one or more denied permissions.
     */
    public static class DeniedPermissionsReceiver extends BroadcastReceiver {
        private PreferenceActivity currentActivity;

        public DeniedPermissionsReceiver(PreferenceActivity activity) {
            this.currentActivity = activity;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            String deniedPermissionsStr = Aware.getSetting(context, Aware_Preferences.DENIED_PERMISSIONS_SERVICES);
            if (!deniedPermissionsStr.equals("")) {
                new ReviewPermissionsDialog(currentActivity).showDialog();
            }
        }
    }

    public static class PermissionListAdapter extends RecyclerView.Adapter<PermissionListAdapter.ViewHolder> {
        private ArrayList<ServicePermissionInfo> permissionList;

        public class ViewHolder extends RecyclerView.ViewHolder {
            public TextView tv_permission_name;
            public TextView tv_permission_description;

            public ViewHolder(View v) {
                super(v);
                tv_permission_name = (TextView) v.findViewById(R.id.permission_name);
                tv_permission_description = (TextView) v.findViewById(R.id.permission_description);
            }
        }

        public PermissionListAdapter(ArrayList<ServicePermissionInfo> permissionList) {
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
            String permissionName = permissionList.get(position).itemName;
            holder.tv_permission_name.setText(permissionName);
            String description = permissionList.get(position).description;
            holder.tv_permission_description.setText(description);
        }

        @Override
        public int getItemCount() {
            return permissionList.size();
        }
    }

    /**
     * Used to display a specific permission and its description.
     */
    private static class ServicePermissionInfo {
        public String itemName;
        public String description;

        public ServicePermissionInfo(String itemName, String description) {
            this.itemName = itemName;
            this.description = description;
        }
    }

    /**
     * Used to display services that are affected by a specific permission.
     */
    private static class PermissionServicesInfo {
        public String permissionName;
        public HashMap<String, String> servicePreferenceMap;

        public PermissionServicesInfo(String permissionName, HashMap<String, String> servicePreferenceMap) {
            this.permissionName = permissionName;
            this.servicePreferenceMap = servicePreferenceMap;
        }
    }

    /**
     * Used to display a list of permissions with one or more associated services.
     */
    public static class PermissionServiceListAdapter extends RecyclerView.Adapter<PermissionServiceListAdapter.ViewHolder> {
        private ArrayList<PermissionServicesInfo> permissionList;

        public class ViewHolder extends RecyclerView.ViewHolder {
            public TextView tv_permission_name;
            public TextView tv_services;

            public ViewHolder(View v) {
                super(v);
                tv_permission_name = (TextView) v.findViewById(R.id.permission_name);
                tv_services = (TextView) v.findViewById(R.id.services);
            }
        }

        public PermissionServiceListAdapter(ArrayList<PermissionServicesInfo> permissionList) {
            this.permissionList = permissionList;
        }

        @Override
        public PermissionServiceListAdapter.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.service_list_item, parent, false);

            ViewHolder vh = new ViewHolder(v);
            return vh;
        }

        @Override
        public void onBindViewHolder(ViewHolder holder, final int position) {
            String permissionName = permissionList.get(position).permissionName;
            // Get readable description of permission
            String[] permissionDescriptions = PERMISSION_DESCRIPTIONS.get(permissionName);
            if (permissionDescriptions != null) {
                holder.tv_permission_name.setText(permissionDescriptions[0]);

                HashMap<String, String> settingPreferences = permissionList.get(position).servicePreferenceMap;
                ArrayList<String> settings = new ArrayList<>(settingPreferences.keySet());

                // Display affected services as bullet points
                SpannableStringBuilder sbBuilder = new SpannableStringBuilder();
                for (int i = 0; i < settings.size(); i++) {
                    String currentSetting = settings.get(i);
                    Spannable itemSpan = new SpannableString(currentSetting + (i < settings.size()-1 ? "\n" : ""));
                    itemSpan.setSpan(new BulletSpan(15), 0, itemSpan.length(), Spanned.SPAN_INCLUSIVE_EXCLUSIVE);
                    sbBuilder.append(itemSpan);
                }
                holder.tv_services.setText(sbBuilder);
            }

        }

        @Override
        public int getItemCount() {
            return permissionList.size();
        }
    }

    /**
     * Manages dialog to prompt reviewing of one or more denied permissions that may affect one or more services.
     * NOTE: Used during bulk activation.
     */
    public static class ReviewPermissionsDialog extends DialogFragment {
        private Activity mActivity;
        private Context context;
        private PermissionServiceListAdapter serviceListAdapter;

        private ArrayList<PermissionServicesInfo> permissionServicesInfos = new ArrayList<>();

        public ReviewPermissionsDialog(Activity activity) {
            this.mActivity = activity;
            this.context = mActivity.getApplicationContext();
        }

        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            AlertDialog.Builder builder = new AlertDialog.Builder(mActivity);
            LayoutInflater inflater = mActivity.getLayoutInflater();
            final View dialogView = inflater.inflate(R.layout.review_permission_dialog, null);
            ArrayList<String> allDeniedPermissions = new ArrayList<>();

            try {
                String deniedPermissionsStr = Aware.getSetting(context, Aware_Preferences.DENIED_PERMISSIONS_SERVICES);
                JSONObject deniedPermissions = new JSONObject(deniedPermissionsStr);
                Iterator<String> permissionIterator = deniedPermissions.keys();

                while (permissionIterator.hasNext()) {
                    String permission = permissionIterator.next();
                    allDeniedPermissions.add(permission);
                    ArrayList<String> preferences = new ArrayList<>(Arrays.asList(deniedPermissions.getString(permission).split(",")));
                    HashMap<String, String> settingsPreferences = new HashMap<>();
                    for (String pref: preferences) {
                        // Remove service from the pending permission queue
                        String serviceClassName = PermissionUtils.removeServiceFromPermissionQueue(context, pref, false);

                        // Construct preference and corresponding layman description to be displayed
                        // NOTE: If pref is a plugin
                        if (serviceClassName != null) {
                            if (PluginsManager.isInstalled(context, serviceClassName) != null) {
                                serviceClassName = serviceClassName + ".Plugin";
                            }

                            // Access class by name to get preference descriptions
                            try {
                                Class<?> serviceClass = Class.forName(serviceClassName);
                                Object classInstance = serviceClass.newInstance();
                                // HACK: Field name is currently hardcoded
                                Field settingsPermissionsField = classInstance.getClass().getField("SETTINGS_PERMISSIONS");
                                HashMap<String, HashMap<String, String>> settingsPermissionsMap = (HashMap<String, HashMap<String, String>>) settingsPermissionsField.get(classInstance);
                                if (settingsPermissionsMap.get(permission) != null) {
                                    HashMap<String, String> permissionSettings = settingsPermissionsMap.get(permission);
                                    settingsPreferences.putAll(permissionSettings);
                                }
                            } catch (ClassNotFoundException | ClassCastException | NoSuchFieldException |
                                     IllegalAccessException | java.lang.InstantiationException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                    PermissionServicesInfo currInfo = new PermissionServicesInfo(permission, settingsPreferences);
                    permissionServicesInfos.add(currInfo);
                }
            } catch (JSONException e) {
                e.printStackTrace();
            }

            if (!permissionServicesInfos.isEmpty()) {
                RecyclerView permissionsRecyclerView = (RecyclerView) dialogView.findViewById(R.id.rc_permissions);
                RecyclerView.LayoutManager permissionsLayoutManager = new LinearLayoutManager(context);
                permissionsRecyclerView.setLayoutManager(permissionsLayoutManager);

                serviceListAdapter = new PermissionServiceListAdapter(permissionServicesInfos);
                permissionsRecyclerView.setAdapter(serviceListAdapter);
            }

            // Resets state
            Aware.setSetting(context, Aware_Preferences.BULK_SERVICE_ACTIVATION, false);

            builder.setView(dialogView);
            builder.setTitle("Sensing Disabled")
                    .setMessage("One or more types of sensing data are unable to be collected because the following permissions have been denied.")
                    .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                        // Restart the relevant processes to prompt permission requests
                        public void onClick(DialogInterface dialog, int id) {
                            ReviewPermissionsDialog.this.dismiss();
                            resetPermissionStatuses(context, allDeniedPermissions);
                            redirectToLocalSettings(Aware_Preferences.REDIRECTED_TO_LOCAL_PERMISSIONS, context, mActivity);
                        }
                    })
                    .setNegativeButton("No", new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog, int id) {
                            ReviewPermissionsDialog.this.dismiss();
                            // Disable the preferences
                            disableDeniedPermissionPreferences(context, mActivity);
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
