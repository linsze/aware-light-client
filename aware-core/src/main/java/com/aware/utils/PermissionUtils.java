package com.aware.utils;

import android.Manifest;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import com.aware.R;

import java.util.ArrayList;
import java.util.HashMap;

public class PermissionUtils {
    public static HashMap<String, String[]> PERMISSION_DESCRIPTIONS = new HashMap<String, String[]>(){{
        put(Manifest.permission.GET_ACCOUNTS, new String[]{"Access to contacts", "Retrieve account used for data synchronization"});
        put(Manifest.permission.WRITE_EXTERNAL_STORAGE, new String[]{"Access to external storage", "Store data locally"});
        put(Manifest.permission.RECORD_AUDIO, new String[]{"Record audio", "Permission for audio detection"});
        put(Manifest.permission.ACCESS_FINE_LOCATION, new String[]{"Access to location", "Store data locally"});
    }};

    public static HashMap<String, String> SENSOR_METHOD_MAPPINGS = new HashMap<String, String>(){{
        put("Accelerometer", "Access account used for data synchronization");
        put(Manifest.permission.WRITE_EXTERNAL_STORAGE, "Access external storage for local data storing");
    }};

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
}
