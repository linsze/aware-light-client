package com.aware.utils;

import static com.aware.utils.PermissionUtils.PERMISSION_DESCRIPTIONS;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.aware.Accelerometer;
import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.R;
//import com.aware.phone.ui.Aware_Join_Study;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


/**
 * Manages dialog to prompt reviewing of denied permissions.
 */
public class PermissionDialog extends DialogFragment {
    private Activity mActivity;
    private Context context;
    private String service;
    private ArrayList<String> pendingPermissions;
    private PermissionUtils.PermissionListAdapter permissionListAdapter;

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
        String warningText;
        if (isMandatory) {
            warningText = "The application is unable to run because the following permissions are denied.";
        } else {
            String serviceName = service.substring(service.lastIndexOf('.')).replaceAll("_", " ");
            warningText = String.format("%s%s will be disabled because the following permissions are denied.", serviceName.substring(0, 1).toUpperCase(), serviceName.substring(1));
        }

        // Sets up list of denied permissions
        ArrayList<PermissionUtils.PermissionInfo> deniedPermissions = new ArrayList<>();
        for (String p: pendingPermissions) {
            String[] permissionDescriptions = PERMISSION_DESCRIPTIONS.get(p);
            PermissionUtils.PermissionInfo permissionInfo = new PermissionUtils.PermissionInfo(permissionDescriptions[0], permissionDescriptions[1]);
            deniedPermissions.add(permissionInfo);
        }

        RecyclerView permissionsRecyclerView = (RecyclerView) dialogView.findViewById(R.id.service_permissions);
        RecyclerView.LayoutManager permissionsLayoutManager = new LinearLayoutManager(context);
        permissionsRecyclerView.setLayoutManager(permissionsLayoutManager);

        permissionListAdapter = new PermissionUtils.PermissionListAdapter(deniedPermissions);
        permissionsRecyclerView.setAdapter(permissionListAdapter);

        builder.setView(dialogView);
        builder.setTitle("Review Permissions")
                .setMessage(warningText)
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
                                try {
                                    Class sensorClass = Class.forName(service);
                                    Intent sensorIntent = new Intent(context, sensorClass);
                                    context.startService(sensorIntent);
                                } catch (ClassNotFoundException e) {
                                    e.printStackTrace();
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
                                try {
                                    Class sensorClass = Class.forName(service);
                                    Intent sensorIntent = new Intent(context, sensorClass);
                                    context.stopService(sensorIntent);
                                } catch (ClassNotFoundException e) {
                                    e.printStackTrace();
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
}
